# WMSR
### [Paper](https://ieeexplore.ieee.org/document/11187314) | [arxiv](https://arxiv.org/abs/2509.24334)

Pytorch codes for the "Wavelet-Assisted Mamba for Satellite-Derived Sea Surface Temperature Super-Resolution", [link](https://github.com/oucailab/WMSR).


> **Wavelet-Assisted Mamba for Satellite-Derived Sea Surface Temperature Super-Resolution** <br>
> [Wankun Chen](https://arxiv.org/search/eess?searchtype=author&query=Chen,+W), [Feng Gao](https://arxiv.org/search/eess?searchtype=author&query=Gao,+F), [Yanhai Gan](https://arxiv.org/search/eess?searchtype=author&query=Gan,+Y), [Jingchao Cao](https://arxiv.org/search/eess?searchtype=author&query=Cao,+J), [Junyu Dong](https://arxiv.org/search/eess?searchtype=author&query=Dong,+J) and [Qian Du](https://arxiv.org/search/eess?searchtype=author&query=Du,+Q). <br>
> Published in TGRS.

## Datasets & Installation
Please refer to the following simple steps for installation. Datas can be download from the public link [[HYCOM]](https://www.hycom.org/dataserver/gofs-3pt1/reanalysis) + [[OISST]]( https://www.ncei.noaa.gov/products/optimum-interpolation-sst) + [[GHRSST]](https://www.ghrsst.org/ghrsst-data-services/for-sst-data-users).

```
cd datas
python nc2png.py    # nc file to png
python sub.py       # obtain low-resolution data
```

### Display of image data file distribution

```bash
sr_datasets/
├── DIV2K/
│   ├── CS/
│   │   ├── HR01.png
│   │   └── ... 
│   └── LR/
│       ├── X2/
│       │   ├── HR01x2.png
│       │   └── ... (2x downsampled data)
│       ├── X3/ ... (3x downsampled data)
│       └── X4/ ... (4x downsampled data)
└── benchmark/
    ├── HYCOM/
    │   ├── CS/
    │   │   ├── hycom_001.png
    │   │   └── ... 
    │   └── LR/
    │       ├── X2/ (2x downsampled data)
    │       ├── X3/ (3x downsampled data)
    │       └── X4/ (4x downsampled data)
    ├── OISST/  ... 
    └── GHRSST/ ...
```


## Training
```
cd wmsr
python train.py --config ./configs/wmsr_x4.yml
```

## Citation

If you believe that WMSR has been helpful to your research or work, please consider citing the following works:

----------
```Bib
@ARTICLE{11187314,
  author={Chen, Wankun and Gao, Feng and Gan, Yanhai and Cao, Jingchao and Dong, Junyu and Du, Qian},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Wavelet-Assisted Mamba for Satellite-Derived Sea Surface Temperature Super-Resolution}, 
  year={2025},
  volume={63},
  pages={1-12},
  doi={10.1109/TGRS.2025.3616324}}
```
