import numpy as np
import netCDF4 as nc
import os
import glob
import scipy
import scipy.misc
import imageio
def NC_to_tiffs(data,Output_folder):
    nc_data_obj = nc.Dataset(data)
    print(nc_data_obj)
    ndvi_arr = np.asarray(nc_data_obj.variables['level-3_binned_data'][:]) # data type
    ndvi_arr_float = ndvi_arr.astype(float)
    ndvi = ndvi_arr_float
    ndvi[np.isnan(ndvi)] = 0
    ndvi[ndvi[:] < 0] = 0
    ndvi = ndvi[::-1]
    for i in len(ndvi_arr[:,]):
        out_tif_name = Output_folder + '\\'+ data.split('\\')[-1].split('.')[0]  + str(i) + '.png'
        ndvi1 = ndvi[::-1]
        imageio.imsave(out_tif_name, ndvi1)
        del ndvi1
def main():
    Input_folder = './nc file'
    Output_folder = './output folder'
    data_list = glob.glob(Input_folder + '\\*.nc')
    for i in range(len(data_list)):
        data = data_list[i]
        NC_to_tiffs(data,Output_folder)
        print(data + '-----conversion successful')
    print('----all finish----')
main()