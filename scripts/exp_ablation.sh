#!/bin/bash
# 消融实验 - 分析不同超参数的影响

# ============================================
# 消融实验1: Hidden Size 对 LSTM+Att 的影响
# ============================================
echo "=== 消融实验1: Hidden Size ==="

python train.py --model_type lstm --use_attention --hidden_size 128 --embedding_dim 128 \
    --batch_size 256 --num_epoch 15 --early_stop 5 --lr 0.001 --save_dir checkpoints/ablation

python train.py --model_type lstm --use_attention --hidden_size 256 --embedding_dim 256 \
    --batch_size 256 --num_epoch 15 --early_stop 5 --lr 0.001 --save_dir checkpoints/ablation

python train.py --model_type lstm --use_attention --hidden_size 512 --embedding_dim 512 \
    --batch_size 256 --num_epoch 15 --early_stop 5 --lr 0.001 --save_dir checkpoints/ablation

# ============================================
# 消融实验2: Transformer层数的影响
# ============================================
echo "=== 消融实验2: Transformer Layers ==="

python train.py --model_type transformer --num_layers 1 \
    --batch_size 256 --num_epoch 15 --early_stop 5 --lr 0.001 --save_dir checkpoints/ablation

python train.py --model_type transformer --num_layers 2 \
    --batch_size 256 --num_epoch 15 --early_stop 5 --lr 0.001 --save_dir checkpoints/ablation

python train.py --model_type transformer --num_layers 3 \
    --batch_size 256 --num_epoch 15 --early_stop 5 --lr 0.001 --save_dir checkpoints/ablation

python train.py --model_type transformer --num_layers 4 \
    --batch_size 256 --num_epoch 15 --early_stop 5 --lr 0.001 --save_dir checkpoints/ablation

# ============================================
# 消融实验3: Transformer注意力头数的影响
# ============================================
echo "=== 消融实验3: Attention Heads ==="

python train.py --model_type transformer --num_heads 1 \
    --batch_size 256 --num_epoch 15 --early_stop 5 --lr 0.001 --save_dir checkpoints/ablation

python train.py --model_type transformer --num_heads 4 \
    --batch_size 256 --num_epoch 15 --early_stop 5 --lr 0.001 --save_dir checkpoints/ablation

python train.py --model_type transformer --num_heads 8 \
    --batch_size 256 --num_epoch 15 --early_stop 5 --lr 0.001 --save_dir checkpoints/ablation

echo "=== 消融实验完成 ==="
