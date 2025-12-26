#!/bin/bash
# 实验组3: Transformer实验

python train.py --model_type transformer --batch_size 256 --num_epoch 20 --early_stop 5 --lr 0.001 \
    --num_layers 3 --num_heads 8 --d_ff 512 --dropout 0.1
