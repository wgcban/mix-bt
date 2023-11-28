#!/bin/bash
gpu=8
dataset=stl10
arch=resnet50
batch_size=512

timestamp=$(date +"%Y%m%d%H%M%S")
session_name="python_session_$timestamp"
echo ${session_name}
screen -dmS "$session_name"
screen -S "$session_name" -X stuff "conda activate ssl-sug^M"

# cifar-100, no-mixup, 0.0078125
# 128 - bd79m7xg
model_path=results/bd79m7xg/bd79m7xg_0.0078125_128_256_stl10_model_1000.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# 1024 - f0jcqvyv
model_path=results/f0jcqvyv/f0jcqvyv_0.0078125_1024_256_stl10_model_1000.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# 2048 - w49vo4ld
model_path=results/w49vo4ld/w49vo4ld_0.0078125_2048_256_stl10_model_1000.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# 4096 - f7singsp
model_path=results/f7singsp/f7singsp_0.0078125_4096_256_stl10_model_1000.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"

# cifar-100, mixup, 0.0078125
# 128 - x1k33z36
model_path=results/x1k33z36/x1k33z36_0.0078125_128_256_stl10_model_1000.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# 1024 - f041u10n
model_path=results/f041u10n/f041u10n_0.0078125_1024_256_stl10_model_850.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# 2048 - khi1enp0
model_path=results/khi1enp0/khi1enp0_0.0078125_2048_256_stl10_model_900.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# 4096 - vy74lew5
model_path=results/vy74lew5/vy74lew5_0.0078125_4096_256_stl10_model_700.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"

screen -S "$session_name" -X detachs