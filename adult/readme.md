## VFLGAN
#### Train vflgan, the following code will train 100 shadow models on the whole dataset
```train
python GAN_vfl_shadow.py
```
#### Train vflgan, the following code will train 100 shadow models on the whole dataset but the LOO target
```train
python GAN_vfl_LOO_33914.py
python GAN_vfl_LOO_37592.py
```

## DP-VFLGAN
#### Train dp-vflgan, the following code will train 100 shadow models on the whole dataset with epsilon=5 and 10
```train
python DP_5_GAN_vfl_shadow.py 
python DP_10_GAN_vfl_shadow.py 
```
#### Train dp-vflgan, the following code will train 100 shadow models on the whole dataset but the LOO target with epsilon=5 and 10
```train
python DP_5_GAN_vfl_LOO_33914.py
python DP_5_GAN_vfl_LOO_37592.py
python DP_10_GAN_vfl_LOO_33914.py
python DP_10_GAN_vfl_LOO_37592.py
```

## Auditting scheme
#### Remeber to change 'shadow path' and 'ano_path' in the 'main()' function before running the following instructions
