#!/bin/bash

# id=3on0l4wl #default barlow twins
# id=ysdrcd38 #mixed barlow twins with x1 loss scale 
id=l418b9zw #mixed barlow twins with all reduce (loss scale: 0.0025) running on viu10
# id=13awtq23 #mixed barlow twins with all reduce (loss scale: 0.1) running on viu3
# id=4xe7mfe3 #mixed barlow twins with x(1/10) loss scale

timestamp=$(date +"%Y%m%d%H%M%S")
session_name="python_session_$timestamp"
echo ${session_name}
screen -dmS "$session_name"
screen -S "$session_name" -X stuff "conda activate ssl-aug^M"
screen -S "$session_name" -X stuff "NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate_imagenet.py /data/wbandar1/datasets/imagenet1k/ /mnt/store/wbandar1/projects/ssl-aug-artifacts/${id}/resnet50.pth --lr-classifier 0.3^M"
screen -S "$session_name" -X detachs