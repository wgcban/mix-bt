#!/bin/bash
path_to_imagenet_data=datasets/imagenet1k/
path_to_model=checkpoints/13awtq23_0.0051_8192_1024_imagenet_0.1_resnet50.pth

timestamp=$(date +"%Y%m%d%H%M%S")
session_name="python_session_$timestamp"
echo ${session_name}
screen -dmS "$session_name"
screen -S "$session_name" -X stuff "conda activate ssl-aug^M"
screen -S "$session_name" -X stuff "NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate_imagenet.py ${path_to_imagenet_data} ${path_to_model} --lr-classifier 0.3^M"
screen -S "$session_name" -X detachs