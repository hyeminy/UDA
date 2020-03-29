#!/bin/bash
python C:\\Users\\9live\\hm_code\\platform_pytorch\\cdan_train.py \
    --data C:\\Users\\9live\\hm_code\\open_data\\office31\\amazon\\images \
    --arch vgg16
    --sobel \
    --clustering Kmeans \
    --nmb_cluster 31 \
    --lr 0.05 \
    --wd -5 \
    --reassign 1. \
    --workers 4 \
    --epochs 200 \
    --start_epoch 0 \
    --batch 256 \
    --momentum 0.9 \
    --checkpoints 25000 \
    --seed 31 \
    --exp C:\\Users\\9live\\hm_code\\UDA\\Deepclustering\\output \
    --verbose
