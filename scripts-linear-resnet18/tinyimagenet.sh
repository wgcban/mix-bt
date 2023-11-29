#!/bin/bash
gpu=0
dataset=tiny_imagenet
arch=resnet18
batch_size=512
model_path=checkpoints/02azq6fs_0.0009765_1024_256_tiny_imagenet_model.pth

timestamp=$(date +"%Y%m%d%H%M%S")
session_name="python_session_$timestamp"
echo ${session_name}
screen -dmS "$session_name"
screen -S "$session_name" -X stuff "conda activate ssl-sug^M"
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
screen -S "$session_name" -X detachs