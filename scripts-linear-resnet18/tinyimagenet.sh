#!/bin/bash
gpu=9
dataset=tiny_imagenet
arch=resnet18
batch_size=512

timestamp=$(date +"%Y%m%d%H%M%S")
session_name="python_session_$timestamp"
echo ${session_name}
screen -dmS "$session_name"
screen -S "$session_name" -X stuff "conda activate ssl-aug^M"

# cifar-100, no-mixup, 0.0078125
# 1024 - bt - er7erhjp
model_path=results/er7erhjp/er7erhjp_0.0009765_1024_256_tiny_imagenet_model.pth
# model_path=results/er7erhjp/er7erhjp_0.0009765_1024_256_tiny_imagenet_model_1000.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"

# mbt-d1024-s4(1/d)-lr0.01 - 02azq6fs
# 1000 epochs
model_path=results/02azq6fs/02azq6fs_0.0009765_1024_256_tiny_imagenet_model_1000.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# 2000 epochs
model_path=results/02azq6fs/02azq6fs_0.0009765_1024_256_tiny_imagenet_model_2000.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# best
model_path=results/02azq6fs/02azq6fs_0.0009765_1024_256_tiny_imagenet_model.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"

# mbt-d1024-s5(1/d)-lr0.01 - rvluiv1h
# 1000 epochs
model_path=results/rvluiv1h/rvluiv1h_0.0009765_1024_256_tiny_imagenet_model_1000.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# 2000 epochs
model_path=results/rvluiv1h/rvluiv1h_0.0009765_1024_256_tiny_imagenet_model_2000.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# best
model_path=results/rvluiv1h/rvluiv1h_0.0009765_1024_256_tiny_imagenet_model.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"

screen -S "$session_name" -X detachs