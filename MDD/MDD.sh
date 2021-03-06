#!/bin/bash
python MDD/mdd_train.py \
    --source_path  \
    --target_path  \
    --init_lr 0.004 \
    --optim_type sgd \
    --lr 0.004 \
    --momentum 0.9 \
    --weight_decay 0.0005 \
    --nesterov True \
    --lr_type inv \
    --gamma 0.001 \
    --decay_rate 0.75 \
    --dataset Office-31 \
    --class_num 31 \
    --width 1024 \
    --srcweight 4 \
    --is_cen False \
    --base_net resnet50 \
    --batch_size 32 \
    --resize_size 256 \
    --crop_size 224 \
    --max_iter 100000 \
    --eval_iter 1000 \
    --output_path