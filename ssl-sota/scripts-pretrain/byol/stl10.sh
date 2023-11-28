#!/bin/bash
gpu=9
dataset=stl10
method=byol
model=resnet50
eval_every=5

timestamp=$(date +"%Y%m%d%H%M%S")
session_name="python_session_$timestamp"
echo ${session_name}
screen -dmS "$session_name"
screen -S "$session_name" -X stuff "conda activate ssl-aug^M"
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python -m train --dataset ${dataset} --epoch 1000 --lr 3e-3 --emb 64 --method ${method} --arch ${model} --eval_every ${eval_every}^M"
screen -S "$session_name" -X detachs