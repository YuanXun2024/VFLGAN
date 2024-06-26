## VFLGAN
#### Train vflgan, the following code will train 100 shadow models on the whole dataset
```train
python GAN_vfl_shadow.py
```
#### Train vflgan, the following code will train 100 shadow models on the whole dataset but the LOO target
```train
python GAN_vfl_LOO_151.py
python GAN_vfl_LOO_1235.py
```

## DP-VFLGAN
#### Train dp-vflgan, the following code will train 100 shadow models on the whole dataset with epsilon=5 and 10
```train
python DP_5_GAN_vfl_shadow.py 
python DP_10_GAN_vfl_shadow.py 
```
#### Train dp-vflgan, the following code will train 100 shadow models on the whole dataset but the LOO target with epsilon=5 and 10
```train
python DP_5_GAN_vfl_LOO_151.py
python DP_5_GAN_vfl_LOO_1235.py
python DP_10_GAN_vfl_LOO_151.py
python DP_10_GAN_vfl_LOO_1235.py
```

## Auditting scheme

#### Generate synthetic datasets (remember to change the parameter folders)
```train
python generate_shadow_data.py
```
#### Generate intermediate features (remember to change the parameter folders)
```train
python generate_IF_data.py
```


#### Remeber to change 'shadow path' and 'ano_path' in the 'main()' function before running the following instructions

#### MIA on the synthetic dataset
```train
python mia.py (correlation feature)
python mia_naive.py (naive feature)
```
#### MIA on the intermediate features
```train
python mia_IF_corr.py (correlation feature)
python mia_IF_naive.py (naive feature)
```