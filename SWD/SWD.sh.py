#!/bin/bash
python C:\Users\9live\hm_code\pytorch_code\UDA\MCD\swd_train.py \
    --proj_dim 128 \
    --batch_size 32 \
    --test_batch_size 32\
    --epochs 50 \
    --lr 0.001 \
    --momentum 0.9 \
    --optimizer momentum \
    --cuda True \
    --seed 1 \
    --log_interval 50 \
    --num_k 4 \
    --num_layer 2 \
    --name output \
    --save_path save\mcd \
    --source_path C:\\Users\\9live\\hm_code\\open_data\\office31\\amazon\\images \
    --target_path C:\\Users\\9live\\hm_code\\open_data\\office31\\dslr\\images \
    --resnet 50