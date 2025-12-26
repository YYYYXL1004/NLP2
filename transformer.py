"""
Transformer Model for Machine Translation

基于nn.Linear()实现的Transformer模型，不调用nn.Transformer。
包含PositionalEncoding、MultiHeadAttention、FFN、Encoder和Decoder层。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from torch import Tensor


# ==================== Positional Encoding ====================

class PositionalEncoding(nn.Module):
    """
    位置编码
    
    使用sin/cos函数为序列中的每个位置生成唯一的位置编码。
    
    公式：
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    其中pos是位置，i是维度索引，d_model是模型维度。
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度（embedding维度）
            max_len: 最大序列长度
            dropout: dropout概率
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # 位置索引 (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算除数项: 10000^(2i/d_model)
        # 使用log空间计算以提高数值稳定性
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数维度使用sin，奇数维度使用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加batch维度 (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # 注册为buffer（不参与梯度计算，但会随模型保存/加载）
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        为输入添加位置编码
        
        Args:
            x: (batch, seq_len, d_model) 输入embedding
        
        Returns:
            (batch, seq_len, d_model) 添加位置编码后的结果
        """
        seq_len = x.size(1)
        # 取出对应长度的位置编码并添加到输入
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)



# ==================== Multi-Head Attention ====================

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    基于nn.Linear实现Q、K、V投影和scaled dot-product attention。
    
    公式：
        Attention(Q, K, V) = softmax(QK^T / √d_k) · V
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
        head_i = Attention(Q·W_Q^i, K·W_K^i, V·W_V^i)
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: dropout概率
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # Q、K、V的线性投影层
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        
        # 输出投影层
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(p=dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: Tensor, key: Tensor, value: Tensor, 
                mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        多头注意力前向传播
        
        Args:
            query: (batch, seq_len_q, d_model) 查询
            key: (batch, seq_len_k, d_model) 键
            value: (batch, seq_len_v, d_model) 值（通常seq_len_k == seq_len_v）
            mask: (batch, 1, seq_len_q, seq_len_k) 或 (batch, 1, 1, seq_len_k) 注意力掩码
        
        Returns:
            output: (batch, seq_len_q, d_model) 注意力输出
            attn_weights: (batch, num_heads, seq_len_q, seq_len_k) 注意力权重
        """
        batch_size = query.size(0)
        
        # 线性投影: (batch, seq_len, d_model)
        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)
        
        # 分割成多头: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        # scores: (batch, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 应用掩码（将被掩码的位置设为负无穷，softmax后接近0）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 注意力权重: (batch, num_heads, seq_len_q, seq_len_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和: (batch, num_heads, seq_len_q, d_k)
        context = torch.matmul(attn_weights, V)
        
        # 合并多头: (batch, num_heads, seq_len_q, d_k) -> (batch, seq_len_q, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 输出投影
        output = self.W_O(context)
        
        return output, attn_weights



# ==================== Position-wise Feed-Forward Network ====================

class PositionwiseFFN(nn.Module):
    """
    位置前馈网络
    
    两层线性变换，中间使用ReLU激活。
    
    公式：
        FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度（输入和输出维度）
            d_ff: 前馈网络隐藏层维度
            dropout: dropout概率
        """
        super(PositionwiseFFN, self).__init__()
        
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前馈网络前向传播
        
        Args:
            x: (batch, seq_len, d_model) 输入
        
        Returns:
            (batch, seq_len, d_model) 输出
        """
        # 第一层线性变换 + ReLU
        hidden = F.relu(self.W_1(x))
        hidden = self.dropout(hidden)
        # 第二层线性变换
        output = self.W_2(hidden)
        return output



# ==================== Transformer Encoder Layer ====================

class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层
    
    包含Self-Attention + FFN + LayerNorm + Residual连接。
    
    结构：
        x -> Self-Attention -> Add & Norm -> FFN -> Add & Norm -> output
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: dropout概率
        """
        super(TransformerEncoderLayer, self).__init__()
        
        # Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Position-wise FFN
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
    
    def forward(self, x: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """
        编码器层前向传播
        
        Args:
            x: (batch, src_len, d_model) 输入
            src_mask: (batch, 1, 1, src_len) 源序列掩码（用于padding）
        
        Returns:
            (batch, src_len, d_model) 输出
        """
        # Self-Attention + Residual + LayerNorm
        attn_output, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # FFN + Residual + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        
        return x



# ==================== Transformer Decoder Layer ====================

class TransformerDecoderLayer(nn.Module):
    """
    Transformer解码器层
    
    包含Masked Self-Attention + Cross-Attention + FFN + LayerNorm + Residual连接。
    
    结构：
        x -> Masked Self-Attention -> Add & Norm 
          -> Cross-Attention (with encoder output) -> Add & Norm 
          -> FFN -> Add & Norm -> output
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: dropout概率
        """
        super(TransformerDecoderLayer, self).__init__()
        
        # Masked Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Cross-Attention (encoder-decoder attention)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Position-wise FFN
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)
    
    def forward(self, x: Tensor, encoder_output: Tensor,
                tgt_mask: Optional[Tensor] = None,
                src_mask: Optional[Tensor] = None) -> Tensor:
        """
        解码器层前向传播
        
        Args:
            x: (batch, tgt_len, d_model) 解码器输入
            encoder_output: (batch, src_len, d_model) 编码器输出
            tgt_mask: (batch, 1, tgt_len, tgt_len) 目标序列掩码（因果掩码 + padding掩码）
            src_mask: (batch, 1, 1, src_len) 源序列掩码（用于cross-attention的padding）
        
        Returns:
            (batch, tgt_len, d_model) 输出
        """
        # Masked Self-Attention + Residual + LayerNorm
        self_attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))
        
        # Cross-Attention + Residual + LayerNorm
        cross_attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))
        
        # FFN + Residual + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))
        
        return x



# ==================== Transformer Seq2Seq ====================

class TransformerSeq2Seq(nn.Module):
    """
    Transformer Seq2Seq模型
    
    组合Encoder和Decoder，实现完整的序列到序列翻译。
    基于nn.Linear()实现，不调用nn.Transformer。
    """
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 d_model: int = 256, num_heads: int = 8, num_layers: int = 3,
                 d_ff: int = 512, dropout: float = 0.1, max_len: int = 100,
                 pad_idx: int = 3, bos_idx: int = 0, eos_idx: int = 1):
        """
        Args:
            src_vocab_size: 源语言词表大小
            tgt_vocab_size: 目标语言词表大小
            d_model: 模型维度
            num_heads: 注意力头数
            num_layers: 编码器/解码器层数
            d_ff: 前馈网络隐藏层维度
            dropout: dropout概率
            max_len: 最大序列长度
            pad_idx: padding符号索引
            bos_idx: BOS符号索引
            eos_idx: EOS符号索引
        """
        super(TransformerSeq2Seq, self).__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.max_len = max_len
        
        # Embedding层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Decoder层
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_src_mask(self, src: Tensor) -> Tensor:
        """
        生成源序列的padding掩码
        
        Args:
            src: (batch, src_len) 源序列
        
        Returns:
            (batch, 1, 1, src_len) padding掩码，padding位置为0，其他为1
        """
        # (batch, src_len) -> (batch, 1, 1, src_len)
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def generate_tgt_mask(self, tgt: Tensor) -> Tensor:
        """
        生成目标序列的掩码（因果掩码 + padding掩码）
        
        Args:
            tgt: (batch, tgt_len) 目标序列
        
        Returns:
            (batch, 1, tgt_len, tgt_len) 组合掩码
        """
        batch_size, tgt_len = tgt.size()
        device = tgt.device
        
        # Padding掩码: (batch, 1, 1, tgt_len)
        padding_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # 因果掩码（下三角矩阵）: (1, 1, tgt_len, tgt_len)
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).unsqueeze(0).unsqueeze(0)
        
        # 组合掩码: (batch, 1, tgt_len, tgt_len)
        # padding_mask广播到(batch, 1, tgt_len, tgt_len)
        tgt_mask = padding_mask & (causal_mask.bool())
        
        return tgt_mask
    
    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        编码器前向传播
        
        Args:
            src: (batch, src_len) 源序列
            src_mask: (batch, 1, 1, src_len) 源序列掩码
        
        Returns:
            (batch, src_len, d_model) 编码器输出
        """
        # Embedding + 缩放 + 位置编码
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # 通过所有编码器层
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return x
    
    def decode(self, tgt: Tensor, encoder_output: Tensor,
               tgt_mask: Tensor, src_mask: Tensor) -> Tensor:
        """
        解码器前向传播
        
        Args:
            tgt: (batch, tgt_len) 目标序列
            encoder_output: (batch, src_len, d_model) 编码器输出
            tgt_mask: (batch, 1, tgt_len, tgt_len) 目标序列掩码
            src_mask: (batch, 1, 1, src_len) 源序列掩码
        
        Returns:
            (batch, tgt_len, d_model) 解码器输出
        """
        # Embedding + 缩放 + 位置编码
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # 通过所有解码器层
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        
        return x
    
    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """
        训练时的前向传播
        
        Args:
            src: (batch, src_len) 源序列
            tgt: (batch, tgt_len) 目标序列（包含BOS，用于teacher forcing）
        
        Returns:
            (batch, tgt_len, vocab_size) 预测的log概率
        """
        # 生成掩码
        src_mask = self.generate_src_mask(src)
        tgt_mask = self.generate_tgt_mask(tgt)
        
        # 编码
        encoder_output = self.encode(src, src_mask)
        
        # 解码
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        
        # 输出投影 + LogSoftmax
        logits = self.output_projection(decoder_output)
        output = F.log_softmax(logits, dim=-1)
        
        return output
    
    def predict(self, src: Tensor) -> Tensor:
        """
        预测（自回归解码）
        
        Args:
            src: (batch, src_len) 源序列
        
        Returns:
            (batch, pred_len) 预测的单词索引序列
        """
        batch_size = src.size(0)
        device = src.device
        
        # 生成源序列掩码并编码
        src_mask = self.generate_src_mask(src)
        encoder_output = self.encode(src, src_mask)
        
        # 初始化目标序列为BOS
        tgt = torch.full((batch_size, 1), self.bos_idx, dtype=torch.long, device=device)
        
        # 自回归解码
        for _ in range(self.max_len - 1):
            # 生成目标掩码
            tgt_mask = self.generate_tgt_mask(tgt)
            
            # 解码
            decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
            
            # 获取最后一个位置的预测
            logits = self.output_projection(decoder_output[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            
            # 拼接到目标序列
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # 检查是否所有序列都生成了EOS
            if (next_token == self.eos_idx).all():
                break
        
        return tgt
