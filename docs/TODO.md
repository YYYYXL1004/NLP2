# NLP期末作业 - 待办清单

## 进度概览

| 模块 | 状态 | BLEU |
|------|------|------|
| RNN (baseline) | ✅ 已完成 | - |
| RNN+Attention | ✅ 已完成 | - |
| LSTM | ✅ 已完成 | - |
| LSTM+Attention | ✅ 已完成 | - |
| GRU | ✅ 已完成 | - |
| GRU+Attention | ✅ 已完成 | - |
| Transformer | ✅ 已完成 | - |

---

## 任务清单

### 阶段1: 项目结构
- [x] 创建`seq2seq_models.py`，包含所有RNN变体模型
- [x] 创建`utils.py`，实现TrainingLogger和Visualizer
- [x] 创建`transformer.py`，实现Transformer模型
- [x] 创建`train.py`，整合训练脚本

### 阶段2: LSTM实现
- [x] 实现LSTMCell (基于nn.Linear，包含4个门)
- [x] 实现EncoderLSTM
- [x] 实现DecoderLSTM

### 阶段3: GRU实现
- [x] 实现GRUCell (基于nn.Linear，包含2个门)
- [x] 实现EncoderGRU
- [x] 实现DecoderGRU

### 阶段4: Attention机制
- [x] 实现Attention类 (Bahdanau加性注意力)
- [x] DecoderLSTMWithAttention
- [x] DecoderGRUWithAttention
- [x] DecoderRNNWithAttention
- [x] Seq2SeqLSTMWithAttention
- [x] Seq2SeqGRUWithAttention
- [x] Seq2SeqRNNWithAttention

### 阶段5: Transformer
- [x] PositionalEncoding (sin/cos位置编码)
- [x] MultiHeadAttention (多头注意力)
- [x] PositionwiseFFN (前馈网络)
- [x] TransformerEncoderLayer
- [x] TransformerDecoderLayer
- [x] TransformerSeq2Seq (完整模型)

### 阶段6: 整合与测试
- [x] 创建`train.py`主脚本
- [x] 命令行参数支持 (argparse)
- [x] 日志记录 (TrainingLogger)
- [x] 可视化曲线 (Visualizer)
- [x] CPU/GPU兼容
- [x] 测试脚本 (test_rnn_variants.py, test_transformer.py)

### 阶段7: 实验与报告
- [ ] 运行所有模型实验
- [ ] 记录实验结果
- [ ] 撰写研究报告

---

## 代码实现说明

### LSTMCell实现要点
基于nn.Linear实现，包含4个门控机制：
- 遗忘门(forget gate): `f_t = σ(W_f·[h,x])`
- 输入门(input gate): `i_t = σ(W_i·[h,x])`
- 候选记忆: `g_t = tanh(W_g·[h,x])`
- 输出门(output gate): `o_t = σ(W_o·[h,x])`
- 记忆更新: `c_t = f_t*c_{t-1} + i_t*g_t`
- 隐状态: `h_t = o_t*tanh(c_t)`

### GRUCell实现要点
基于nn.Linear实现，包含2个门控机制：
- 重置门(reset gate): `r_t = σ(W_r·[h,x])`
- 更新门(update gate): `z_t = σ(W_z·[h,x])`
- 候选隐状态: `n_t = tanh(W_n·[r_t*h, x])`
- 隐状态更新: `h_t = (1-z_t)*h_{t-1} + z_t*n_t`

### Attention实现要点
Bahdanau加性注意力：
- 注意力分数: `e = v^T·tanh(W_h·h_enc + W_s·h_dec)`
- 注意力权重: `α = softmax(e)`
- 上下文向量: `c = Σα·h_enc`

### Transformer实现要点
- 位置编码: sin/cos函数
- 多头注意力: Q,K,V投影 + scaled dot-product
- 因果掩码: 下三角矩阵防止看到未来
- 残差连接 + LayerNorm

---

## 实验记录

### 实验环境
- Python版本: 3.x
- PyTorch版本: 2.x
- 设备: CPU / GPU
- 训练数据量: 26,187条

### 实验结果

| 模型 | 训练时间 | 最终Loss | 验证BLEU | 测试BLEU |
|------|----------|----------|----------|----------|
| RNN | | | | |
| RNN+Att | | | | |
| LSTM | | | | |
| LSTM+Att | | | | |
| GRU | | | | |
| GRU+Att | | | | |
| Transformer | | | | |

---

## 问题记录

### 问题1: [待记录]
**日期**: 
**描述**: 
**解决方案**: 

---

## 思考题分析

### LSTM/GRU如何实现长距离依赖？

**LSTM**:
- 遗忘门(forget gate): 控制丢弃多少旧信息
- 输入门(input gate): 控制接收多少新信息
- 记忆单元(cell state): 信息可以沿着cell state直接传递，避免梯度消失
- 输出门(output gate): 控制输出多少信息

**GRU**:
- 更新门(update gate): 控制保留多少旧信息
- 重置门(reset gate): 控制忽略多少旧信息
- 比LSTM参数更少，但效果相近

**关键机制**: 门控机制让梯度可以直接流过，避免了普通RNN的梯度消失问题。

---

## 参考资料

- LSTM原理: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- Seq2Seq+Attention: https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html
- Transformer: https://theaisummer.com/transformer/

---

## 实验设计

### 实验目标
1. 验证 LSTM/GRU 相比普通 RNN 的性能提升
2. 验证 Attention 机制对翻译质量的提升效果
3. 验证 Transformer 架构的优越性
4. 分析不同模型的训练效率与资源消耗

### 公共超参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 256 | 批次大小 |
| num_epoch | 20 | 训练轮数 |
| early_stop | 5 | 早停patience |
| learning_rate | 0.001 | 学习率 |
| hidden_size | 256 | 隐藏层维度 |
| embed_size | 256 | 词嵌入维度 |
| max_len | 10 | 最大序列长度 |
| train_data | 26,187 | 训练集大小 |

### Transformer 专用配置

| 参数 | 值 | 说明 |
|------|-----|------|
| num_layer | 3 | 编码器/解码器层数 |
| num_head | 8 | 注意力头数 |
| hidden_size | 256 | 隐藏层维度 |
| ffn_hidden_size | 512 | FFN中间层维度 |
| dropout | 0.1 | Dropout比率 |

---

### 实验组1: RNN变体对比 (无Attention)

**目的**: 对比普通RNN、LSTM、GRU三种循环单元的性能差异

| 实验编号 | 模型 | 命令 |
|----------|------|------|
| Exp1-1 | RNN | `python train.py --model_type rnn --batch_size 256 --num_epoch 20 --early_stop 5 --lr 0.001` |
| Exp1-2 | LSTM | `python train.py --model_type lstm --batch_size 256 --num_epoch 20 --early_stop 5 --lr 0.001` |
| Exp1-3 | GRU | `python train.py --model_type gru --batch_size 256 --num_epoch 20 --early_stop 5 --lr 0.001` |

**结果记录**:

| 实验编号 | 模型 | 训练时间 | 最终Loss | 验证BLEU | 测试BLEU | 参考BLEU |
|----------|------|----------|----------|----------|----------|----------|
| Exp1-1 | RNN | | | | | 1.41 |
| Exp1-2 | LSTM | | | | | - |
| Exp1-3 | GRU | | | | | - |

**预期**: LSTM ≈ GRU > RNN

---

### 实验组2: Attention机制效果对比

**目的**: 验证Attention机制对各RNN变体的提升效果

| 实验编号 | 模型 | 命令 |
|----------|------|------|
| Exp2-1 | RNN+Att | `python train.py --model_type rnn --use_attention --batch_size 256 --num_epoch 20 --early_stop 5 --lr 0.001` |
| Exp2-2 | LSTM+Att | `python train.py --model_type lstm --use_attention --batch_size 256 --num_epoch 20 --early_stop 5 --lr 0.001` |
| Exp2-3 | GRU+Att | `python train.py --model_type gru --use_attention --batch_size 256 --num_epoch 20 --early_stop 5 --lr 0.001` |

**结果记录**:

| 实验编号 | 模型 | 训练时间 | 最终Loss | 验证BLEU | 测试BLEU | 参考BLEU |
|----------|------|----------|----------|----------|----------|----------|
| Exp2-1 | RNN+Att | | | | | 13.15 |
| Exp2-2 | LSTM+Att | | | | | 13.52 |
| Exp2-3 | GRU+Att | | | | | - |

**预期**: LSTM+Att ≈ GRU+Att > RNN+Att >> 无Attention版本

---

### 实验组3: Transformer实验

**目的**: 验证Transformer架构的性能优势

| 实验编号 | 模型 | 命令 |
|----------|------|------|
| Exp3-1 | Transformer | `python train.py --model_type transformer --batch_size 256 --num_epoch 20 --early_stop 5 --lr 0.001 --num_layers 3 --num_heads 8 --d_ff 512 --dropout 0.1` |

**结果记录**:

| 实验编号 | 模型 | 训练时间 | 最终Loss | 验证BLEU | 测试BLEU | 参考BLEU |
|----------|------|----------|----------|----------|----------|----------|
| Exp3-1 | Transformer | | | | | 23.41 |

**预期**: Transformer >> LSTM+Att > RNN+Att

---

### 实验组4: 综合对比汇总

**目的**: 汇总所有模型的最终结果，进行横向对比

| 排名 | 模型 | 测试BLEU | 训练时间(GPU) | 训练时间(CPU) | GPU显存 | 相对提升 |
|------|------|----------|---------------|---------------|---------|----------|
| 1 | Transformer | | ~5.5min | ~1h10min | ~1501MB | baseline |
| 2 | LSTM+Att | | ~3.1min | ~1h10min | ~1449MB | |
| 3 | GRU+Att | | | | | |
| 4 | RNN+Att | | ~2.4min | ~1h10min | ~1431MB | |
| 5 | LSTM | | | | | |
| 6 | GRU | | | | | |
| 7 | RNN | | ~1.5min | ~50min | ~1249MB | |

---

### 实验执行顺序

建议按以下顺序执行实验，便于对比分析：

```bash
# 实验组1: RNN变体对比
bash scripts/exp1_rnn_variants.sh

# 实验组2: Attention机制效果
bash scripts/exp2_attention.sh

# 实验组3: Transformer
bash scripts/exp3_transformer.sh
```

---

### 分析要点

1. **RNN vs LSTM/GRU**: 门控机制对长距离依赖的影响
2. **有无Attention**: 注意力机制对翻译对齐的帮助
3. **RNN系列 vs Transformer**: 并行计算与自注意力的优势
4. **训练效率**: 不同模型的时间/显存开销对比
5. **收敛速度**: Loss下降曲线对比分析

