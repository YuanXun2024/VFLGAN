## VFLGAN
#### Train vflgan
```train
python WGAN_GP_real_VFL_v2.py --train
```
#### Generate samples of vflgan
```inference
python WGAN_GP_real_VFL_v2.py
```
#### Draw IS curve of training process of vflgan
```IS curve
python evaluation_IS.py
```
#### Draw FID curve of training process of vflgan
```FID curve
python evaluation_fid.py
```

## DP-VFLGAN
#### Train dp-vflgan (10,1e-5)-DP
```train
python WGAN_GP_real_VFL_v2.py --train --sigma 0.802
```
#### Generate samples of dp-vflgan
```inference
python WGAN_GP_real_VFL_v2.py
```
#### Draw FID curve of training process of dp-vflgan
```FID curve
python evaluation_fid_dp.py
```