import math
import argparse, yaml
import utils
import os
from tqdm import tqdm
import logging
import sys
import time
import importlib
import glob
from torchvision.transforms import Compose
import torch.nn.parallel
from focal_frequency_loss import FocalFrequencyLoss as FFL
import cv2
ffl = FFL(loss_weight=1.0, alpha=1.0) 
CUDA_VISIBLE_DEVICES=0
parser = argparse.ArgumentParser(description='WMSR')
## yaml configuration files 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
parser.add_argument('--config', type=str, default='./configs/wmsr_x4.yml', help = 'pre-config file for training')
parser.add_argument('--resume', type=str, default=None, help = 'resume training or not')
import random
from datetime import datetime

import matplotlib.image as img
if __name__ == '__main__':
    args = parser.parse_args()
    if args.config:
       opt = vars(args)
       yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
       opt.update(yaml_args)
    ## set visibel gpu   
    gpu_ids_str = str(args.gpu_ids).replace('[','').replace(']','')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_ids_str)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import MultiStepLR, StepLR
    from datas.utils import create_datasets


    ## select active gpu devices
    device = None
    if args.gpu_ids is not None and torch.cuda.is_available():
        print('use cuda & cudnn for acceleration!')
        print('the gpu id is: {}'.format(args.gpu_ids))
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        print('use cpu for training!')
        device = torch.device('cpu')
    torch.set_num_threads(args.threads)

    ## create dataset for training and validating
    train_dataloader, valid_dataloaders = create_datasets(args)

    ## definitions of model
    try:
        model = utils.import_module('models.{}_network'.format(args.model, args.model)).create_model(args)
    except Exception:
        raise ValueError('not supported model type! or something')
    model = nn.DataParallel(model).to(device)

    ## definition of loss and optimizer
    loss_func = nn.L1Loss()#nn.SmoothL1Loss()#
    loss_func1 = nn.MSELoss()
    loss_func2 = nn.SmoothL1Loss()
    loss_func3 = FFL(loss_weight=1.0, alpha=1.0) 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.decays, gamma=args.gamma)

    ## load pretrain
    #if args.pretrain is not None:
    #print('load pretrained model: {}!'.format(args.pretrain))
    ckpt = torch.load('./2/model_x2_4.pt')#args.pretrain)#
    model.load_state_dict(ckpt['model_state_dict'])
    
    ## resume training
    start_epoch = 1
    if args.resume is not None:
        ckpt_files = glob.glob(os.path.join(args.resume, 'models', "*.pt"))
        if len(ckpt_files) != 0:
            ckpt_files = sorted(ckpt_files, key=lambda x: int(x.replace('.pt','').split('_')[-1]))
            #ckpt = torch.load(ckpt_files[-1])
            ckpt = torch.load('./2/model_x2_4.pt')
            prev_epoch = ckpt['epoch']

            start_epoch = prev_epoch + 1
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            stat_dict = ckpt['stat_dict']
            ## reset folder and param
            experiment_path = args.resume
            log_name = os.path.join(experiment_path, 'log.txt')
            experiment_model_path = os.path.join(experiment_path, 'models')
            print('select {}, resume training from epoch {}.'.format(ckpt_files[-1], start_epoch))
    else:
        ## auto-generate the output logname
        experiment_name = None
        timestamp = utils.cur_timestamp_str()
        if args.log_name is None:
            experiment_name = '{}-{}-x{}-{}'.format(args.model, 'fp32', args.scale, timestamp)
        else:
            experiment_name = '{}-{}'.format(args.log_name, timestamp)
        experiment_path = os.path.join(args.log_path, experiment_name)
        log_name = os.path.join(experiment_path, 'log.txt')
        stat_dict = utils.get_stat_dict()
        ## create folder for ckpt and stat
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        experiment_model_path = os.path.join(experiment_path, 'models')
        if not os.path.exists(experiment_model_path):
            os.makedirs(experiment_model_path)
        ## save training paramters
        exp_params = vars(args)
        exp_params_name = os.path.join(experiment_path, 'config.yml')
        with open(exp_params_name, 'w') as exp_params_file:
            yaml.dump(exp_params, exp_params_file, default_flow_style=False)

    sys.stdout = utils.ExperimentLogger(log_name, sys.stdout)
    #print(model)
    sys.stdout.flush()
    resume = 1
    ## start training
    timer_start = time.time()
    if resume == 1:
        if resume == 1:
            torch.set_grad_enabled(False)
            test_log = ''
            model = model.eval()

            for valid_dataloader in valid_dataloaders:
                avg_psnr, avg_ssim, avg_rmse = 0.0, 0.0, 0.0
                name = valid_dataloader['name']
                print(name)
                loader = valid_dataloader['dataloader']
                for lr, hr in tqdm(loader, ncols=80):
                    lr, hr = lr.to(device), hr.to(device)
                    with torch.no_grad():
                        sr = model(lr)
                    time_str = datetime.strftime(datetime.now(), '%Y-%m%d-%H-%M-%S')
                    srr = sr.detach().cpu().numpy()
                    srr = srr[0,:,:,:]
                    srr = srr.transpose(1,2,0)
                    hr = hr.clamp(0, 255)
                    sr = sr.clamp(0, 255)

                    cv2.imwrite('hdfhg.png',cv2.cvtColor(srr, cv2.COLOR_RGB2BGR))
                     # conver to ycbcr
                    if args.colors == 3:
                        hr_ycbcr = utils.rgb_to_ycbcr(hr)
                        sr_ycbcr = utils.rgb_to_ycbcr(sr)
                        hr = hr_ycbcr[:, 0:1, :, :]
                        sr = sr_ycbcr[:, 0:1, :, :]
                    # crop image for evaluation 
                    hr = hr[:, :, args.scale:-args.scale, args.scale:-args.scale]
                    sr = sr[:, :, args.scale:-args.scale, args.scale:-args.scale]
                    print(sr.shape)
                    # calculate psnr and ssim
                    psnr = utils.calc_psnr(sr, hr)       
                    ssim = utils.calc_ssim(sr, hr)  
                    rmse = utils.calc_rmse(sr, hr)
                    avg_psnr += psnr
                    avg_ssim += ssim
                    avg_rmse += rmse
                epoch = 1
                avg_psnr = round(avg_psnr/len(loader) + 5e-5, 2)
                avg_ssim = round(avg_ssim/len(loader) + 5e-5, 4)
                avg_rmse = round(avg_rmse/len(loader) + 5e-5, 3)
                stat_dict[name]['psnrs'].append(avg_psnr)
                stat_dict[name]['ssims'].append(avg_ssim)
                stat_dict[name]['rmses'].append(avg_rmse)
                if stat_dict[name]['best_psnr']['value'] < avg_psnr:
                    stat_dict[name]['best_psnr']['value'] = avg_psnr
                    stat_dict[name]['best_psnr']['epoch'] = epoch
                if stat_dict[name]['best_ssim']['value'] < avg_ssim:
                    stat_dict[name]['best_ssim']['value'] = avg_ssim
                    stat_dict[name]['best_ssim']['epoch'] = epoch
                if stat_dict[name]['best_rmse']['value'] < avg_rmse:
                    stat_dict[name]['best_rmse']['value'] = avg_rmse
                    stat_dict[name]['best_rmse']['epoch'] = epoch
                test_log += '[{}-X{}], PSNR/SSIM: {:.2f}/{:.4f}/{:.3f} (Best: {:.2f}/{:.4f}/{:.3f}, Epoch: {}/{}/{})\n'.format(
                    name, args.scale, float(avg_psnr), float(avg_ssim), float(avg_rmse), 
                    stat_dict[name]['best_psnr']['value'], stat_dict[name]['best_rmse']['value'], stat_dict[name]['best_ssim']['value'],
                    stat_dict[name]['best_psnr']['epoch'], stat_dict[name]['best_rmse']['epoch'], stat_dict[name]['best_ssim']['value'])           
            print('test_log')
            print(test_log)
            sys.stdout.flush()
            stat_dict_name = os.path.join(experiment_path, 'stat_dict.yml')
            with open(stat_dict_name, 'w') as stat_dict_file:
                yaml.dump(stat_dict, stat_dict_file, default_flow_style=False)
        scheduler.step()
        
