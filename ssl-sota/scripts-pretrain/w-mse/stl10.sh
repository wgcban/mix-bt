#!/bin/bash
gpu=9
dataset=stl10
method=w_mse
model=resnet50

timestamp=$(date +"%Y%m%d%H%M%S")
session_name="python_session_$timestamp"
echo ${session_name}
screen -dmS "$session_name"
screen -S "$session_name" -X stuff "conda activate ssl-aug^M"
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python -m train --dataset ${dataset} --epoch 1000 --lr 2e-3 --bs 256 --emb 128 --w_size 256 --w_eps 10e-3 --method ${method} --arch ${model}^M"
screen -S "$session_name" -X detachs