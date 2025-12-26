# NLP期末作业 - 待办清单

## 进度概览

| 模块 | 状态 | BLEU |
|------|------|------|
| RNN (baseline) | ✅ 已有 | 1.41 |
| RNN+Attention | ⬜ 待实现 | - |
| LSTM+Attention | ⬜ 待实现 | - |
| GRU+Attention | ⬜ 待实现 | - |
| Transformer | ⬜ 待实现 | - |

---

## 任务清单

### 阶段1: 项目结构
- [ ] 创建`seq2seq_models.py`，复制基础代码
- [ ] 创建`utils.py`，实现Logger和Visualizer

### 阶段2: LSTM实现
- [ ] 实现LSTMCell (基于nn.Linear)
- [ ] 实现EncoderLSTM
- [ ] 实现DecoderLSTM

### 阶段3: GRU实现
- [ ] 实现GRUCell (基于nn.Linear)
- [ ] 实现EncoderGRU
- [ ] 实现DecoderGRU

### 阶段4: Attention机制
- [ ] 实现Attention类
- [ ] DecoderLSTM + Attention
- [ ] DecoderGRU + Attention
- [ ] DecoderRNN + Attention

### 阶段5: Transformer
- [ ] PositionalEncoding
- [ ] MultiHeadAttention
- [ ] PositionwiseFFN
- [ ] TransformerEncoderLayer
- [ ] TransformerDecoderLayer
- [ ] TransformerSeq2Seq

### 阶段6: 整合与测试
- [ ] 创建`train.py`主脚本
- [ ] 命令行参数支持
- [ ] 日志记录
- [ ] 可视化曲线
- [ ] CPU/GPU兼容测试

---

## 实验记录

### 实验环境
- Python版本: 
- PyTorch版本: 
- 设备: CPU / GPU (型号)
- 训练数据量: 26,187条

### 实验结果

| 模型 | 训练时间 | 最终Loss | 验证BLEU | 测试BLEU |
|------|----------|----------|----------|----------|
| RNN | | | | |
| RNN+Att | | | | |
| LSTM+Att | | | | |
| GRU+Att | | | | |
| Transformer | | | | |

---

## 问题记录

### 问题1: [标题]
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
