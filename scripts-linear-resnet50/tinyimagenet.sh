#!/bin/bash
gpu=7
dataset=tiny_imagenet
arch=resnet50
batch_size=512
epochs=500

timestamp=$(date +"%Y%m%d%H%M%S")
session_name="python_session_$timestamp"
echo ${session_name}
screen -dmS "$session_name"
screen -S "$session_name" -X stuff "conda activate ssl-aug^M"

# # cifar-100, no-mixup, 0.0078125
# # 128 - 44xgm1r0
# model_path=results/44xgm1r0/44xgm1r0_0.0078125_128_256_tiny_imagenet_model_1000.pth
# screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# # 1024 - h2a9qv17
# model_path=results/h2a9qv17/h2a9qv17_0.0078125_1024_256_tiny_imagenet_model_1000.pth
# screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# # 2048 - ao3puap4
# model_path=results/ao3puap4/ao3puap4_0.0078125_2048_256_tiny_imagenet_model_1000.pth
# screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# # 4096 - bn125mik
# model_path=results/bn125mik/bn125mik_0.0078125_4096_256_tiny_imagenet_model_1000.pth
# screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"

# # cifar-100, mixup, 0.0078125
# # 128 - m9m8k16h
# model_path=results/m9m8k16h/m9m8k16h_0.0078125_128_256_tiny_imagenet_model_1000.pth
# screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# # 1024 - wzrzb7c9
# model_path=results/wzrzb7c9/wzrzb7c9_0.0078125_1024_256_tiny_imagenet_model_1000.pth
# screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# # 2048 - b4viw7r1
# model_path=results/b4viw7r1/b4viw7r1_0.0078125_2048_256_tiny_imagenet_model_1000.pth
# screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# # 4096
# # model_path=results/jf6lt1z0/jf6lt1z0_0.0078125_4096_256_tiny_imagenet_model_1000.pth
# # screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path}^M"

# With cosine-shedule
# mbt-d1024-s4(1/d): kxlkigsv
# 1000 epochs
model_path=results/kxlkigsv/kxlkigsv_0.0009765_1024_256_tiny_imagenet_model_1000.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch} --epochs ${epochs}^M"

# 2000 epochs
model_path=results/kxlkigsv/kxlkigsv_0.0009765_1024_256_tiny_imagenet_model_2000.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch} --epochs ${epochs}^M"

# best epoch
model_path=results/kxlkigsv/kxlkigsv_0.0009765_1024_256_tiny_imagenet_model.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch} --epochs ${epochs}^M"


# bt-d1024-s(1/d): nqklilow #
model_path=results/nqklilow/nqklilow_0.0009765_1024_256_tiny_imagenet_model_1000.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch} --epochs ${epochs}^M"

# 2000 epochs
model_path=results/nqklilow/nqklilow_0.0009765_1024_256_tiny_imagenet_model_2000.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch} --epochs ${epochs}^M"

# best epoch
model_path=results/nqklilow/nqklilow_0.0009765_1024_256_tiny_imagenet_model.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch} --epochs ${epochs}^M"


screen -S "$session_name" -X detachs