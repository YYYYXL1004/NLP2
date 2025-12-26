#!/bin/bash
# 实验组2: Attention机制效果对比

# RNN + Attention
python train.py --model_type rnn --use_attention --batch_size 256 --num_epoch 20 --early_stop 5 --lr 0.001 --exp_name rnn_attn

# LSTM + Attention
python train.py --model_type lstm --use_attention --batch_size 256 --num_epoch 20 --early_stop 5 --lr 0.001 --exp_name lstm_attn

# GRU + Attention
python train.py --model_type gru --use_attention --batch_size 256 --num_epoch 20 --early_stop 5 --lr 0.001 --exp_name gru_attn
