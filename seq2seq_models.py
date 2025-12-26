"""
Seq2Seq Models for Machine Translation

基于参考代码实现的seq2seq模型，包含RNN、LSTM、GRU变体及Attention机制。
所有RNN变体基于nn.Linear()实现，不调用PyTorch内置的LSTM/GRU模块。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
from torch import Tensor


# ==================== 数据加载 ====================

def load_data(num_train: int = -1):
    """
    加载中英翻译数据集
    
    Args:
        num_train: 训练集大小，-1表示使用全部数据
    
    Returns:
        zh_sents: 中文句子字典，包含train/val/test
        en_sents: 英文句子字典，包含train/val/test
    """
    zh_sents = {}
    en_sents = {}
    for split in ['train', 'val', 'test']:
        zh_sents[split] = []
        en_sents[split] = []
        with open(f"data/zh_en_{split}.txt", encoding='utf-8') as f:
            for line in f.readlines():
                zh, en = line.strip().split("\t")
                zh = zh.split()
                en = en.split()
                zh_sents[split].append(zh)
                en_sents[split].append(en)
    num_train = len(zh_sents['train']) if num_train == -1 else num_train
    zh_sents['train'] = zh_sents['train'][:num_train]
    en_sents['train'] = en_sents['train'][:num_train]
    print("训练集 验证集 测试集大小分别为", len(zh_sents['train']), len(zh_sents['val']), len(zh_sents['test']))
    return zh_sents, en_sents


# ==================== 词表 ====================

class Vocab:
    """词表类，用于单词和索引之间的转换"""
    
    def __init__(self):
        self.word2idx = {}
        self.word2cnt = {}
        self.idx2word = []
        self.add_word("[BOS]")
        self.add_word("[EOS]")
        self.add_word("[UNK]")
        self.add_word("[PAD]")
    
    def add_word(self, word: str):
        """将单词word加入到词表中"""
        if word not in self.word2idx:
            self.word2cnt[word] = 0
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        self.word2cnt[word] += 1
    
    def add_sent(self, sent: List[str]):
        """将句子sent中的每一个单词加入到词表中"""
        for word in sent:
            self.add_word(word)
    
    def index(self, word: str) -> int:
        """若word在词表中则返回其下标，否则返回[UNK]对应序号"""
        return self.word2idx.get(word, self.word2idx["[UNK]"])
    
    def encode(self, sent: List[str], max_len: int) -> List[int]:
        """在句子sent的首尾分别添加BOS和EOS之后编码为整数序列"""
        encoded = [self.word2idx["[BOS]"]] + [self.index(word) for word in sent][:max_len] + [self.word2idx["[EOS]"]]
        return encoded
    
    def decode(self, encoded: List[int], strip_bos_eos_pad: bool = False) -> List[str]:
        """将整数序列解码为单词序列"""
        return [self.idx2word[_] for _ in encoded 
                if not strip_bos_eos_pad or self.idx2word[_] not in ["[BOS]", "[EOS]", "[PAD]"]]
    
    def __len__(self) -> int:
        """返回词表大小"""
        return len(self.idx2word)


# ==================== RNN Cell (Baseline) ====================

class RNNCell(nn.Module):
    """基础RNN单元，基于nn.Linear实现"""
    
    def __init__(self, input_size: int, hidden_size: int):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Linear(input_size, hidden_size)
        self.weight_hh = nn.Linear(hidden_size, hidden_size)

    def forward(self, input: Tensor, hx: Tensor) -> Tensor:
        """
        Args:
            input: (batch, input_size)
            hx: (batch, hidden_size)
        
        Returns:
            h_new: (batch, hidden_size)
        """
        igates = self.weight_ih(input)
        hgates = self.weight_hh(hx)
        ret = torch.tanh(igates + hgates)
        return ret


# ==================== LSTM Cell ====================

class LSTMCell(nn.Module):
    """
    LSTM单元，基于nn.Linear实现
    
    LSTM通过门控机制解决RNN的梯度消失问题，实现长距离依赖。
    
    公式：
        f_t = σ(W_f · [h_{t-1}, x_t] + b_f)    # 遗忘门
        i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    # 输入门
        c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c) # 候选记忆
        c_t = f_t * c_{t-1} + i_t * c̃_t        # 更新记忆
        o_t = σ(W_o · [h_{t-1}, x_t] + b_o)    # 输出门
        h_t = o_t * tanh(c_t)                   # 输出隐状态
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 输入门 (input gate)
        self.W_ii = nn.Linear(input_size, hidden_size)
        self.W_hi = nn.Linear(hidden_size, hidden_size)
        
        # 遗忘门 (forget gate)
        self.W_if = nn.Linear(input_size, hidden_size)
        self.W_hf = nn.Linear(hidden_size, hidden_size)
        
        # 候选记忆 (cell gate / candidate)
        self.W_ig = nn.Linear(input_size, hidden_size)
        self.W_hg = nn.Linear(hidden_size, hidden_size)
        
        # 输出门 (output gate)
        self.W_io = nn.Linear(input_size, hidden_size)
        self.W_ho = nn.Linear(hidden_size, hidden_size)

    def forward(self, input: Tensor, hx: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Args:
            input: (batch, input_size) 当前时刻输入
            hx: (h, c) 元组
                h: (batch, hidden_size) 上一时刻隐状态
                c: (batch, hidden_size) 上一时刻记忆状态
        
        Returns:
            (h_new, c_new): 元组
                h_new: (batch, hidden_size) 更新后的隐状态
                c_new: (batch, hidden_size) 更新后的记忆状态
        """
        h, c = hx
        
        # 输入门: i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
        i_t = torch.sigmoid(self.W_ii(input) + self.W_hi(h))
        
        # 遗忘门: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
        f_t = torch.sigmoid(self.W_if(input) + self.W_hf(h))
        
        # 候选记忆: c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
        g_t = torch.tanh(self.W_ig(input) + self.W_hg(h))
        
        # 输出门: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
        o_t = torch.sigmoid(self.W_io(input) + self.W_ho(h))
        
        # 更新记忆: c_t = f_t * c_{t-1} + i_t * c̃_t
        c_new = f_t * c + i_t * g_t
        
        # 输出隐状态: h_t = o_t * tanh(c_t)
        h_new = o_t * torch.tanh(c_new)
        
        return h_new, c_new


# ==================== GRU Cell ====================

class GRUCell(nn.Module):
    """
    GRU单元，基于nn.Linear实现
    
    GRU是LSTM的简化版本，只有两个门，参数更少。
    
    公式：
        r_t = σ(W_r · [h_{t-1}, x_t] + b_r)    # 重置门
        z_t = σ(W_z · [h_{t-1}, x_t] + b_z)    # 更新门
        h̃_t = tanh(W_h · [r_t * h_{t-1}, x_t] + b_h)  # 候选隐状态
        h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t  # 更新隐状态
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 重置门 (reset gate)
        self.W_ir = nn.Linear(input_size, hidden_size)
        self.W_hr = nn.Linear(hidden_size, hidden_size)
        
        # 更新门 (update gate)
        self.W_iz = nn.Linear(input_size, hidden_size)
        self.W_hz = nn.Linear(hidden_size, hidden_size)
        
        # 候选隐状态 (candidate hidden state)
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_hn = nn.Linear(hidden_size, hidden_size)

    def forward(self, input: Tensor, hx: Tensor) -> Tensor:
        """
        Args:
            input: (batch, input_size) 当前时刻输入
            hx: (batch, hidden_size) 上一时刻隐状态
        
        Returns:
            h_new: (batch, hidden_size) 更新后的隐状态
        """
        # 重置门: r_t = σ(W_r · [h_{t-1}, x_t] + b_r)
        r_t = torch.sigmoid(self.W_ir(input) + self.W_hr(hx))
        
        # 更新门: z_t = σ(W_z · [h_{t-1}, x_t] + b_z)
        z_t = torch.sigmoid(self.W_iz(input) + self.W_hz(hx))
        
        # 候选隐状态: h̃_t = tanh(W_h · [r_t * h_{t-1}, x_t] + b_h)
        n_t = torch.tanh(self.W_in(input) + self.W_hn(r_t * hx))
        
        # 更新隐状态: h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
        h_new = (1 - z_t) * hx + z_t * n_t
        
        return h_new


# ==================== Attention ====================

class Attention(nn.Module):
    """
    Bahdanau Attention（加性注意力）
    
    让decoder在每个时间步关注encoder不同位置的信息。
    
    公式：
        e_{ij} = v^T · tanh(W_h · h_j + W_s · s_i)  # 注意力分数
        α_{ij} = softmax(e_{ij})                     # 注意力权重
        c_i = Σ α_{ij} · h_j                         # context向量
    """
    
    def __init__(self, hidden_size: int):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        
        # W_h: 用于encoder隐状态的线性变换
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        # W_s: 用于decoder隐状态的线性变换
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)
        # v: 用于计算最终注意力分数的向量
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, decoder_hidden: Tensor, encoder_hiddens: Tensor) -> Tuple[Tensor, Tensor]:
        """
        计算注意力权重和context向量
        
        Args:
            decoder_hidden: (batch, hidden_size) 当前decoder隐状态
            encoder_hiddens: (batch, src_len, hidden_size) 所有encoder隐状态
        
        Returns:
            context: (batch, hidden_size) 加权求和后的context向量
            attn_weights: (batch, src_len) 注意力权重（用于可视化）
        """
        batch_size, src_len, _ = encoder_hiddens.size()
        
        # 扩展decoder_hidden以便与encoder_hiddens计算
        # decoder_hidden: (batch, hidden_size) -> (batch, 1, hidden_size)
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1)
        
        # 计算注意力分数
        # W_h · h_j: (batch, src_len, hidden_size)
        encoder_transformed = self.W_h(encoder_hiddens)
        # W_s · s_i: (batch, 1, hidden_size)
        decoder_transformed = self.W_s(decoder_hidden_expanded)
        
        # e_{ij} = v^T · tanh(W_h · h_j + W_s · s_i)
        # (batch, src_len, hidden_size) -> (batch, src_len, 1) -> (batch, src_len)
        energy = self.v(torch.tanh(encoder_transformed + decoder_transformed)).squeeze(-1)
        
        # α_{ij} = softmax(e_{ij})
        # (batch, src_len)
        attn_weights = F.softmax(energy, dim=-1)
        
        # c_i = Σ α_{ij} · h_j
        # (batch, 1, src_len) @ (batch, src_len, hidden_size) -> (batch, 1, hidden_size)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_hiddens).squeeze(1)
        
        return context, attn_weights


# ==================== RNN Encoder/Decoder (Baseline) ====================

class EncoderRNN(nn.Module):
    """RNN编码器"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = RNNCell(embedding_dim, hidden_size)
    
    def forward(self, input: Tensor, hidden: Tensor) -> Tensor:
        """
        Args:
            input: (batch,) 输入单词索引
            hidden: (batch, hidden_size) 上一时刻隐状态
        
        Returns:
            hidden: (batch, hidden_size) 更新后的隐状态
        """
        embedding = self.embed(input)
        hidden = self.rnn(embedding, hidden)
        return hidden


# ==================== LSTM Encoder ====================

class EncoderLSTM(nn.Module):
    """LSTM编码器"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super(EncoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = LSTMCell(embedding_dim, hidden_size)
    
    def forward(self, input: Tensor, hx: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Args:
            input: (batch,) 输入单词索引
            hx: (h, c) 元组
                h: (batch, hidden_size) 上一时刻隐状态
                c: (batch, hidden_size) 上一时刻记忆状态
        
        Returns:
            (h_new, c_new): 更新后的(隐状态, 记忆状态)
        """
        embedding = self.embed(input)
        h_new, c_new = self.lstm(embedding, hx)
        return h_new, c_new


# ==================== GRU Encoder ====================

class EncoderGRU(nn.Module):
    """GRU编码器"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super(EncoderGRU, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.gru = GRUCell(embedding_dim, hidden_size)
    
    def forward(self, input: Tensor, hidden: Tensor) -> Tensor:
        """
        Args:
            input: (batch,) 输入单词索引
            hidden: (batch, hidden_size) 上一时刻隐状态
        
        Returns:
            hidden: (batch, hidden_size) 更新后的隐状态
        """
        embedding = self.embed(input)
        hidden = self.gru(embedding, hidden)
        return hidden


class DecoderRNN(nn.Module):
    """RNN解码器（无Attention）"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = RNNCell(embedding_dim, hidden_size)
        self.h2o = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, input: Tensor, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            input: (batch,) 输入单词索引
            hidden: (batch, hidden_size) 上一时刻隐状态
        
        Returns:
            output: (batch, vocab_size) 预测的log概率
            hidden: (batch, hidden_size) 更新后的隐状态
        """
        embedding = self.embed(input)
        hidden = self.rnn(embedding, hidden)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden


# ==================== LSTM Decoder ====================

class DecoderLSTM(nn.Module):
    """LSTM解码器（无Attention）"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super(DecoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = LSTMCell(embedding_dim, hidden_size)
        self.h2o = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, input: Tensor, hx: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
            input: (batch,) 输入单词索引
            hx: (h, c) 元组
                h: (batch, hidden_size) 上一时刻隐状态
                c: (batch, hidden_size) 上一时刻记忆状态
        
        Returns:
            output: (batch, vocab_size) 预测的log概率
            (h_new, c_new): 更新后的(隐状态, 记忆状态)
        """
        embedding = self.embed(input)
        h_new, c_new = self.lstm(embedding, hx)
        output = self.h2o(h_new)
        output = self.softmax(output)
        return output, (h_new, c_new)


# ==================== GRU Decoder ====================

class DecoderGRU(nn.Module):
    """GRU解码器（无Attention）"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super(DecoderGRU, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.gru = GRUCell(embedding_dim, hidden_size)
        self.h2o = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, input: Tensor, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            input: (batch,) 输入单词索引
            hidden: (batch, hidden_size) 上一时刻隐状态
        
        Returns:
            output: (batch, vocab_size) 预测的log概率
            hidden: (batch, hidden_size) 更新后的隐状态
        """
        embedding = self.embed(input)
        hidden = self.gru(embedding, hidden)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden


# ==================== Decoder with Attention ====================

class DecoderRNNWithAttention(nn.Module):
    """RNN解码器（带Attention）"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super(DecoderRNNWithAttention, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.attention = Attention(hidden_size)
        # RNN输入为embedding + context
        self.rnn = RNNCell(embedding_dim + hidden_size, hidden_size)
        # 输出层输入为hidden + context
        self.h2o = nn.Linear(hidden_size * 2, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, input: Tensor, hidden: Tensor, 
                encoder_hiddens: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            input: (batch,) 输入单词索引
            hidden: (batch, hidden_size) 上一时刻隐状态
            encoder_hiddens: (batch, src_len, hidden_size) 所有encoder隐状态
        
        Returns:
            output: (batch, vocab_size) 预测的log概率
            hidden: (batch, hidden_size) 更新后的隐状态
            attn_weights: (batch, src_len) 注意力权重
        """
        embedding = self.embed(input)  # (batch, embedding_dim)
        
        # 计算attention
        context, attn_weights = self.attention(hidden, encoder_hiddens)
        
        # 将embedding和context拼接作为RNN输入
        rnn_input = torch.cat([embedding, context], dim=-1)  # (batch, embedding_dim + hidden_size)
        hidden = self.rnn(rnn_input, hidden)
        
        # 将hidden和context拼接用于预测
        output_input = torch.cat([hidden, context], dim=-1)  # (batch, hidden_size * 2)
        output = self.h2o(output_input)
        output = self.softmax(output)
        
        return output, hidden, attn_weights


class DecoderLSTMWithAttention(nn.Module):
    """LSTM解码器（带Attention）"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super(DecoderLSTMWithAttention, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.attention = Attention(hidden_size)
        # LSTM输入为embedding + context
        self.lstm = LSTMCell(embedding_dim + hidden_size, hidden_size)
        # 输出层输入为hidden + context
        self.h2o = nn.Linear(hidden_size * 2, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, input: Tensor, hx: Tuple[Tensor, Tensor], 
                encoder_hiddens: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor]:
        """
        Args:
            input: (batch,) 输入单词索引
            hx: (h, c) 元组
                h: (batch, hidden_size) 上一时刻隐状态
                c: (batch, hidden_size) 上一时刻记忆状态
            encoder_hiddens: (batch, src_len, hidden_size) 所有encoder隐状态
        
        Returns:
            output: (batch, vocab_size) 预测的log概率
            (h_new, c_new): 更新后的(隐状态, 记忆状态)
            attn_weights: (batch, src_len) 注意力权重
        """
        h, c = hx
        embedding = self.embed(input)  # (batch, embedding_dim)
        
        # 计算attention（使用当前hidden状态）
        context, attn_weights = self.attention(h, encoder_hiddens)
        
        # 将embedding和context拼接作为LSTM输入
        lstm_input = torch.cat([embedding, context], dim=-1)  # (batch, embedding_dim + hidden_size)
        h_new, c_new = self.lstm(lstm_input, (h, c))
        
        # 将hidden和context拼接用于预测
        output_input = torch.cat([h_new, context], dim=-1)  # (batch, hidden_size * 2)
        output = self.h2o(output_input)
        output = self.softmax(output)
        
        return output, (h_new, c_new), attn_weights


class DecoderGRUWithAttention(nn.Module):
    """GRU解码器（带Attention）"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super(DecoderGRUWithAttention, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.attention = Attention(hidden_size)
        # GRU输入为embedding + context
        self.gru = GRUCell(embedding_dim + hidden_size, hidden_size)
        # 输出层输入为hidden + context
        self.h2o = nn.Linear(hidden_size * 2, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, input: Tensor, hidden: Tensor, 
                encoder_hiddens: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            input: (batch,) 输入单词索引
            hidden: (batch, hidden_size) 上一时刻隐状态
            encoder_hiddens: (batch, src_len, hidden_size) 所有encoder隐状态
        
        Returns:
            output: (batch, vocab_size) 预测的log概率
            hidden: (batch, hidden_size) 更新后的隐状态
            attn_weights: (batch, src_len) 注意力权重
        """
        embedding = self.embed(input)  # (batch, embedding_dim)
        
        # 计算attention
        context, attn_weights = self.attention(hidden, encoder_hiddens)
        
        # 将embedding和context拼接作为GRU输入
        gru_input = torch.cat([embedding, context], dim=-1)  # (batch, embedding_dim + hidden_size)
        hidden = self.gru(gru_input, hidden)
        
        # 将hidden和context拼接用于预测
        output_input = torch.cat([hidden, context], dim=-1)  # (batch, hidden_size * 2)
        output = self.h2o(output_input)
        output = self.softmax(output)
        
        return output, hidden, attn_weights


# ==================== Seq2Seq RNN (Baseline) ====================

class Seq2SeqRNN(nn.Module):
    """基础Seq2Seq模型，使用RNN"""
    
    def __init__(self, src_vocab: Vocab, tgt_vocab: Vocab, 
                 embedding_dim: int, hidden_size: int, max_len: int):
        super(Seq2SeqRNN, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.hidden_size = hidden_size
        self.max_len = max_len
        
        self.encoder = EncoderRNN(len(src_vocab), embedding_dim, hidden_size)
        self.decoder = DecoderRNN(len(tgt_vocab), embedding_dim, hidden_size)
        
    def init_hidden(self, batch_size: int) -> Tensor:
        """初始化编码器端隐状态为全0向量"""
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_size).to(device)
    
    def init_tgt_bos(self, batch_size: int) -> Tensor:
        """预测时，初始化解码器端输入为[BOS]"""
        device = next(self.parameters()).device
        return (torch.ones(batch_size) * self.tgt_vocab.index("[BOS]")).long().to(device)
    
    def forward_encoder(self, src: Tensor) -> Tuple[Tensor, Tensor]:
        """
        编码器前向传播
        
        Args:
            src: (batch, src_len) 源语言序列
        
        Returns:
            hidden: (batch, hidden_size) 最终隐状态
            encoder_hiddens: (batch, src_len, hidden_size) 所有时刻的隐状态
        """
        Bs, Ls = src.size()
        hidden = self.init_hidden(batch_size=Bs)
        encoder_hiddens = []
        
        for i in range(Ls):
            input = src[:, i]
            hidden = self.encoder(input, hidden)
            encoder_hiddens.append(hidden)
        
        encoder_hiddens = torch.stack(encoder_hiddens, dim=1)
        return hidden, encoder_hiddens
    
    def forward_decoder(self, tgt: Tensor, hidden: Tensor, 
                        encoder_hiddens: Optional[Tensor] = None) -> Tensor:
        """
        解码器前向传播（训练时使用teacher forcing）
        
        Args:
            tgt: (batch, tgt_len) 目标语言序列
            hidden: (batch, hidden_size) 编码器最终隐状态
            encoder_hiddens: 未使用，保留接口兼容性
        
        Returns:
            outputs: (batch, tgt_len, vocab_size) 预测结果
        """
        Bs, Lt = tgt.size()
        outputs = []
        
        for i in range(Lt):
            input = tgt[:, i]  # teacher forcing
            output, hidden = self.decoder(input, hidden)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs
        
    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """训练时的前向传播"""
        hidden, encoder_hiddens = self.forward_encoder(src)
        outputs = self.forward_decoder(tgt, hidden, encoder_hiddens)
        return outputs
    
    def predict(self, src: Tensor) -> Tensor:
        """
        预测（自回归解码）
        
        Args:
            src: (1, src_len) 源语言序列
        
        Returns:
            preds: (1, pred_len) 预测的单词索引序列
        """
        hidden, encoder_hiddens = self.forward_encoder(src)
        input = self.init_tgt_bos(batch_size=src.shape[0])
        preds = [input]
        
        while len(preds) < self.max_len:
            output, hidden = self.decoder(input, hidden)
            input = output.argmax(-1)
            preds.append(input)
            if input == self.tgt_vocab.index("[EOS]"):
                break
        
        preds = torch.stack(preds, dim=-1)
        return preds


# ==================== Seq2Seq LSTM ====================

class Seq2SeqLSTM(nn.Module):
    """Seq2Seq模型，使用LSTM"""
    
    def __init__(self, src_vocab: Vocab, tgt_vocab: Vocab, 
                 embedding_dim: int, hidden_size: int, max_len: int):
        super(Seq2SeqLSTM, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.hidden_size = hidden_size
        self.max_len = max_len
        
        self.encoder = EncoderLSTM(len(src_vocab), embedding_dim, hidden_size)
        self.decoder = DecoderLSTM(len(tgt_vocab), embedding_dim, hidden_size)
        
    def init_hidden(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """初始化编码器端隐状态和记忆状态为全0向量"""
        device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.hidden_size).to(device)
        c = torch.zeros(batch_size, self.hidden_size).to(device)
        return h, c
    
    def init_tgt_bos(self, batch_size: int) -> Tensor:
        """预测时，初始化解码器端输入为[BOS]"""
        device = next(self.parameters()).device
        return (torch.ones(batch_size) * self.tgt_vocab.index("[BOS]")).long().to(device)
    
    def forward_encoder(self, src: Tensor) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        """
        编码器前向传播
        
        Args:
            src: (batch, src_len) 源语言序列
        
        Returns:
            (hidden, cell): 最终(隐状态, 记忆状态)
            encoder_hiddens: (batch, src_len, hidden_size) 所有时刻的隐状态
        """
        Bs, Ls = src.size()
        hidden, cell = self.init_hidden(batch_size=Bs)
        encoder_hiddens = []
        
        for i in range(Ls):
            input = src[:, i]
            hidden, cell = self.encoder(input, (hidden, cell))
            encoder_hiddens.append(hidden)
        
        encoder_hiddens = torch.stack(encoder_hiddens, dim=1)
        return (hidden, cell), encoder_hiddens
    
    def forward_decoder(self, tgt: Tensor, hx: Tuple[Tensor, Tensor], 
                        encoder_hiddens: Optional[Tensor] = None) -> Tensor:
        """
        解码器前向传播（训练时使用teacher forcing）
        
        Args:
            tgt: (batch, tgt_len) 目标语言序列
            hx: (hidden, cell) 编码器最终状态
            encoder_hiddens: 未使用，保留接口兼容性（用于Attention）
        
        Returns:
            outputs: (batch, tgt_len, vocab_size) 预测结果
        """
        Bs, Lt = tgt.size()
        outputs = []
        hidden, cell = hx
        
        for i in range(Lt):
            input = tgt[:, i]  # teacher forcing
            output, (hidden, cell) = self.decoder(input, (hidden, cell))
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs
        
    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """训练时的前向传播"""
        hx, encoder_hiddens = self.forward_encoder(src)
        outputs = self.forward_decoder(tgt, hx, encoder_hiddens)
        return outputs
    
    def predict(self, src: Tensor) -> Tensor:
        """
        预测（自回归解码）
        
        Args:
            src: (1, src_len) 源语言序列
        
        Returns:
            preds: (1, pred_len) 预测的单词索引序列
        """
        hx, encoder_hiddens = self.forward_encoder(src)
        hidden, cell = hx
        input = self.init_tgt_bos(batch_size=src.shape[0])
        preds = [input]
        
        while len(preds) < self.max_len:
            output, (hidden, cell) = self.decoder(input, (hidden, cell))
            input = output.argmax(-1)
            preds.append(input)
            if input == self.tgt_vocab.index("[EOS]"):
                break
        
        preds = torch.stack(preds, dim=-1)
        return preds


# ==================== Seq2Seq GRU ====================

class Seq2SeqGRU(nn.Module):
    """Seq2Seq模型，使用GRU"""
    
    def __init__(self, src_vocab: Vocab, tgt_vocab: Vocab, 
                 embedding_dim: int, hidden_size: int, max_len: int):
        super(Seq2SeqGRU, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.hidden_size = hidden_size
        self.max_len = max_len
        
        self.encoder = EncoderGRU(len(src_vocab), embedding_dim, hidden_size)
        self.decoder = DecoderGRU(len(tgt_vocab), embedding_dim, hidden_size)
        
    def init_hidden(self, batch_size: int) -> Tensor:
        """初始化编码器端隐状态为全0向量"""
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_size).to(device)
    
    def init_tgt_bos(self, batch_size: int) -> Tensor:
        """预测时，初始化解码器端输入为[BOS]"""
        device = next(self.parameters()).device
        return (torch.ones(batch_size) * self.tgt_vocab.index("[BOS]")).long().to(device)
    
    def forward_encoder(self, src: Tensor) -> Tuple[Tensor, Tensor]:
        """
        编码器前向传播
        
        Args:
            src: (batch, src_len) 源语言序列
        
        Returns:
            hidden: (batch, hidden_size) 最终隐状态
            encoder_hiddens: (batch, src_len, hidden_size) 所有时刻的隐状态
        """
        Bs, Ls = src.size()
        hidden = self.init_hidden(batch_size=Bs)
        encoder_hiddens = []
        
        for i in range(Ls):
            input = src[:, i]
            hidden = self.encoder(input, hidden)
            encoder_hiddens.append(hidden)
        
        encoder_hiddens = torch.stack(encoder_hiddens, dim=1)
        return hidden, encoder_hiddens
    
    def forward_decoder(self, tgt: Tensor, hidden: Tensor, 
                        encoder_hiddens: Optional[Tensor] = None) -> Tensor:
        """
        解码器前向传播（训练时使用teacher forcing）
        
        Args:
            tgt: (batch, tgt_len) 目标语言序列
            hidden: (batch, hidden_size) 编码器最终隐状态
            encoder_hiddens: 未使用，保留接口兼容性（用于Attention）
        
        Returns:
            outputs: (batch, tgt_len, vocab_size) 预测结果
        """
        Bs, Lt = tgt.size()
        outputs = []
        
        for i in range(Lt):
            input = tgt[:, i]  # teacher forcing
            output, hidden = self.decoder(input, hidden)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs
        
    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """训练时的前向传播"""
        hidden, encoder_hiddens = self.forward_encoder(src)
        outputs = self.forward_decoder(tgt, hidden, encoder_hiddens)
        return outputs
    
    def predict(self, src: Tensor) -> Tensor:
        """
        预测（自回归解码）
        
        Args:
            src: (1, src_len) 源语言序列
        
        Returns:
            preds: (1, pred_len) 预测的单词索引序列
        """
        hidden, encoder_hiddens = self.forward_encoder(src)
        input = self.init_tgt_bos(batch_size=src.shape[0])
        preds = [input]
        
        while len(preds) < self.max_len:
            output, hidden = self.decoder(input, hidden)
            input = output.argmax(-1)
            preds.append(input)
            if input == self.tgt_vocab.index("[EOS]"):
                break
        
        preds = torch.stack(preds, dim=-1)
        return preds


# ==================== Seq2Seq with Attention ====================

class Seq2SeqRNNWithAttention(nn.Module):
    """Seq2Seq模型，使用RNN + Attention"""
    
    def __init__(self, src_vocab: Vocab, tgt_vocab: Vocab, 
                 embedding_dim: int, hidden_size: int, max_len: int):
        super(Seq2SeqRNNWithAttention, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.hidden_size = hidden_size
        self.max_len = max_len
        
        self.encoder = EncoderRNN(len(src_vocab), embedding_dim, hidden_size)
        self.decoder = DecoderRNNWithAttention(len(tgt_vocab), embedding_dim, hidden_size)
        
    def init_hidden(self, batch_size: int) -> Tensor:
        """初始化编码器端隐状态为全0向量"""
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_size).to(device)
    
    def init_tgt_bos(self, batch_size: int) -> Tensor:
        """预测时，初始化解码器端输入为[BOS]"""
        device = next(self.parameters()).device
        return (torch.ones(batch_size) * self.tgt_vocab.index("[BOS]")).long().to(device)
    
    def forward_encoder(self, src: Tensor) -> Tuple[Tensor, Tensor]:
        """
        编码器前向传播
        
        Args:
            src: (batch, src_len) 源语言序列
        
        Returns:
            hidden: (batch, hidden_size) 最终隐状态
            encoder_hiddens: (batch, src_len, hidden_size) 所有时刻的隐状态
        """
        Bs, Ls = src.size()
        hidden = self.init_hidden(batch_size=Bs)
        encoder_hiddens = []
        
        for i in range(Ls):
            input = src[:, i]
            hidden = self.encoder(input, hidden)
            encoder_hiddens.append(hidden)
        
        encoder_hiddens = torch.stack(encoder_hiddens, dim=1)
        return hidden, encoder_hiddens
    
    def forward_decoder(self, tgt: Tensor, hidden: Tensor, 
                        encoder_hiddens: Tensor) -> Tensor:
        """
        解码器前向传播（训练时使用teacher forcing）
        
        Args:
            tgt: (batch, tgt_len) 目标语言序列
            hidden: (batch, hidden_size) 编码器最终隐状态
            encoder_hiddens: (batch, src_len, hidden_size) 所有encoder隐状态
        
        Returns:
            outputs: (batch, tgt_len, vocab_size) 预测结果
        """
        Bs, Lt = tgt.size()
        outputs = []
        
        for i in range(Lt):
            input = tgt[:, i]  # teacher forcing
            output, hidden, _ = self.decoder(input, hidden, encoder_hiddens)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs
        
    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """训练时的前向传播"""
        hidden, encoder_hiddens = self.forward_encoder(src)
        outputs = self.forward_decoder(tgt, hidden, encoder_hiddens)
        return outputs
    
    def predict(self, src: Tensor) -> Tensor:
        """
        预测（自回归解码）
        
        Args:
            src: (1, src_len) 源语言序列
        
        Returns:
            preds: (1, pred_len) 预测的单词索引序列
        """
        hidden, encoder_hiddens = self.forward_encoder(src)
        input = self.init_tgt_bos(batch_size=src.shape[0])
        preds = [input]
        
        while len(preds) < self.max_len:
            output, hidden, _ = self.decoder(input, hidden, encoder_hiddens)
            input = output.argmax(-1)
            preds.append(input)
            if input == self.tgt_vocab.index("[EOS]"):
                break
        
        preds = torch.stack(preds, dim=-1)
        return preds


class Seq2SeqLSTMWithAttention(nn.Module):
    """Seq2Seq模型，使用LSTM + Attention"""
    
    def __init__(self, src_vocab: Vocab, tgt_vocab: Vocab, 
                 embedding_dim: int, hidden_size: int, max_len: int):
        super(Seq2SeqLSTMWithAttention, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.hidden_size = hidden_size
        self.max_len = max_len
        
        self.encoder = EncoderLSTM(len(src_vocab), embedding_dim, hidden_size)
        self.decoder = DecoderLSTMWithAttention(len(tgt_vocab), embedding_dim, hidden_size)
        
    def init_hidden(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """初始化编码器端隐状态和记忆状态为全0向量"""
        device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.hidden_size).to(device)
        c = torch.zeros(batch_size, self.hidden_size).to(device)
        return h, c
    
    def init_tgt_bos(self, batch_size: int) -> Tensor:
        """预测时，初始化解码器端输入为[BOS]"""
        device = next(self.parameters()).device
        return (torch.ones(batch_size) * self.tgt_vocab.index("[BOS]")).long().to(device)
    
    def forward_encoder(self, src: Tensor) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        """
        编码器前向传播
        
        Args:
            src: (batch, src_len) 源语言序列
        
        Returns:
            (hidden, cell): 最终(隐状态, 记忆状态)
            encoder_hiddens: (batch, src_len, hidden_size) 所有时刻的隐状态
        """
        Bs, Ls = src.size()
        hidden, cell = self.init_hidden(batch_size=Bs)
        encoder_hiddens = []
        
        for i in range(Ls):
            input = src[:, i]
            hidden, cell = self.encoder(input, (hidden, cell))
            encoder_hiddens.append(hidden)
        
        encoder_hiddens = torch.stack(encoder_hiddens, dim=1)
        return (hidden, cell), encoder_hiddens
    
    def forward_decoder(self, tgt: Tensor, hx: Tuple[Tensor, Tensor], 
                        encoder_hiddens: Tensor) -> Tensor:
        """
        解码器前向传播（训练时使用teacher forcing）
        
        Args:
            tgt: (batch, tgt_len) 目标语言序列
            hx: (hidden, cell) 编码器最终状态
            encoder_hiddens: (batch, src_len, hidden_size) 所有encoder隐状态
        
        Returns:
            outputs: (batch, tgt_len, vocab_size) 预测结果
        """
        Bs, Lt = tgt.size()
        outputs = []
        hidden, cell = hx
        
        for i in range(Lt):
            input = tgt[:, i]  # teacher forcing
            output, (hidden, cell), _ = self.decoder(input, (hidden, cell), encoder_hiddens)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs
        
    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """训练时的前向传播"""
        hx, encoder_hiddens = self.forward_encoder(src)
        outputs = self.forward_decoder(tgt, hx, encoder_hiddens)
        return outputs
    
    def predict(self, src: Tensor) -> Tensor:
        """
        预测（自回归解码）
        
        Args:
            src: (1, src_len) 源语言序列
        
        Returns:
            preds: (1, pred_len) 预测的单词索引序列
        """
        hx, encoder_hiddens = self.forward_encoder(src)
        hidden, cell = hx
        input = self.init_tgt_bos(batch_size=src.shape[0])
        preds = [input]
        
        while len(preds) < self.max_len:
            output, (hidden, cell), _ = self.decoder(input, (hidden, cell), encoder_hiddens)
            input = output.argmax(-1)
            preds.append(input)
            if input == self.tgt_vocab.index("[EOS]"):
                break
        
        preds = torch.stack(preds, dim=-1)
        return preds


class Seq2SeqGRUWithAttention(nn.Module):
    """Seq2Seq模型，使用GRU + Attention"""
    
    def __init__(self, src_vocab: Vocab, tgt_vocab: Vocab, 
                 embedding_dim: int, hidden_size: int, max_len: int):
        super(Seq2SeqGRUWithAttention, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.hidden_size = hidden_size
        self.max_len = max_len
        
        self.encoder = EncoderGRU(len(src_vocab), embedding_dim, hidden_size)
        self.decoder = DecoderGRUWithAttention(len(tgt_vocab), embedding_dim, hidden_size)
        
    def init_hidden(self, batch_size: int) -> Tensor:
        """初始化编码器端隐状态为全0向量"""
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_size).to(device)
    
    def init_tgt_bos(self, batch_size: int) -> Tensor:
        """预测时，初始化解码器端输入为[BOS]"""
        device = next(self.parameters()).device
        return (torch.ones(batch_size) * self.tgt_vocab.index("[BOS]")).long().to(device)
    
    def forward_encoder(self, src: Tensor) -> Tuple[Tensor, Tensor]:
        """
        编码器前向传播
        
        Args:
            src: (batch, src_len) 源语言序列
        
        Returns:
            hidden: (batch, hidden_size) 最终隐状态
            encoder_hiddens: (batch, src_len, hidden_size) 所有时刻的隐状态
        """
        Bs, Ls = src.size()
        hidden = self.init_hidden(batch_size=Bs)
        encoder_hiddens = []
        
        for i in range(Ls):
            input = src[:, i]
            hidden = self.encoder(input, hidden)
            encoder_hiddens.append(hidden)
        
        encoder_hiddens = torch.stack(encoder_hiddens, dim=1)
        return hidden, encoder_hiddens
    
    def forward_decoder(self, tgt: Tensor, hidden: Tensor, 
                        encoder_hiddens: Tensor) -> Tensor:
        """
        解码器前向传播（训练时使用teacher forcing）
        
        Args:
            tgt: (batch, tgt_len) 目标语言序列
            hidden: (batch, hidden_size) 编码器最终隐状态
            encoder_hiddens: (batch, src_len, hidden_size) 所有encoder隐状态
        
        Returns:
            outputs: (batch, tgt_len, vocab_size) 预测结果
        """
        Bs, Lt = tgt.size()
        outputs = []
        
        for i in range(Lt):
            input = tgt[:, i]  # teacher forcing
            output, hidden, _ = self.decoder(input, hidden, encoder_hiddens)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs
        
    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """训练时的前向传播"""
        hidden, encoder_hiddens = self.forward_encoder(src)
        outputs = self.forward_decoder(tgt, hidden, encoder_hiddens)
        return outputs
    
    def predict(self, src: Tensor) -> Tensor:
        """
        预测（自回归解码）
        
        Args:
            src: (1, src_len) 源语言序列
        
        Returns:
            preds: (1, pred_len) 预测的单词索引序列
        """
        hidden, encoder_hiddens = self.forward_encoder(src)
        input = self.init_tgt_bos(batch_size=src.shape[0])
        preds = [input]
        
        while len(preds) < self.max_len:
            output, hidden, _ = self.decoder(input, hidden, encoder_hiddens)
            input = output.argmax(-1)
            preds.append(input)
            if input == self.tgt_vocab.index("[EOS]"):
                break
        
        preds = torch.stack(preds, dim=-1)
        return preds


# ==================== DataLoader工具函数 ====================

def collate(data_list: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[Tensor, Tensor]:
    """DataLoader的collate函数"""
    src = torch.stack([torch.LongTensor(_[0]) for _ in data_list])
    tgt = torch.stack([torch.LongTensor(_[1]) for _ in data_list])
    return src, tgt


def padding(inp_ids: List[int], max_len: int, pad_id: int) -> np.ndarray:
    """将序列填充到指定长度"""
    max_len += 2  # include [BOS] and [EOS]
    ids_ = np.ones(max_len, dtype=np.int32) * pad_id
    actual_len = min(len(inp_ids), max_len)
    ids_[:actual_len] = inp_ids[:actual_len]
    return ids_


def create_dataloader(zh_sents: dict, en_sents: dict, 
                      zh_vocab: Vocab, en_vocab: Vocab,
                      max_len: int, batch_size: int, pad_id: int):
    """
    创建数据加载器
    
    Args:
        zh_sents: 中文句子字典
        en_sents: 英文句子字典
        zh_vocab: 中文词表
        en_vocab: 英文词表
        max_len: 最大序列长度
        batch_size: 批次大小
        pad_id: 填充符号的索引
    
    Returns:
        trainloader, validloader, testloader
    """
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        shuffle = True if split == 'train' else False
        datas = [
            (padding(zh_vocab.encode(zh, max_len), max_len, pad_id), 
             padding(en_vocab.encode(en, max_len), max_len, pad_id)) 
            for zh, en in zip(zh_sents[split], en_sents[split])
        ]
        dataloaders[split] = torch.utils.data.DataLoader(
            datas, batch_size=batch_size, shuffle=shuffle, collate_fn=collate
        )
    return dataloaders['train'], dataloaders['val'], dataloaders['test']
