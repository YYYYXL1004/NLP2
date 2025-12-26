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

## 命令行使用示例

```bash
# 训练LSTM+Attention
python train.py --model_type lstm --use_attention --num_epoch 10

# 训练GRU+Attention
python train.py --model_type gru --use_attention --num_epoch 10

# 训练Transformer
python train.py --model_type transformer --num_epoch 10

# 运行测试
python test_rnn_variants.py
python test_transformer.py
```

---

## 参考资料

- LSTM原理: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- Seq2Seq+Attention: https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html
- Transformer: https://theaisummer.com/transformer/
