#!/bin/bash
# 实验组1: RNN变体对比 (无Attention)

# RNN
python train.py --model_type rnn --batch_size 256 --num_epoch 20 --early_stop 5 --lr 0.001 --exp_name rnn

# LSTM
python train.py --model_type lstm --batch_size 256 --num_epoch 20 --early_stop 5 --lr 0.001 --exp_name lstm

# GRU
python train.py --model_type gru --batch_size 256 --num_epoch 20 --early_stop 5 --lr 0.001 --exp_name gru
