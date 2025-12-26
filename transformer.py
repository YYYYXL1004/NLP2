"""
Transformer模型 - 机器翻译
基于nn.Linear实现，不调用nn.Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """位置编码: sin/cos函数"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)  # 偶数维sin
        pe[:, 1::2] = torch.cos(pos * div)  # 奇数维cos
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """多头注意力: Q,K,V分头计算后拼接"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        
        # 线性变换后分头
        Q = self.W_Q(q).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(k).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(v).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V)
        
        # 拼接多头
        out = out.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.W_O(out), attn

class FeedForward(nn.Module):
    """前馈网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    """编码器层: Self-Attn -> Add&Norm -> FFN -> Add&Norm"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))  # 残差连接
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x

class DecoderLayer(nn.Module):
    """解码器层: Masked Self-Attn -> Cross-Attn -> FFN"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        # masked self-attention
        attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        # cross-attention (看encoder输出)
        attn_out, _ = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout2(attn_out))
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        return x

class TransformerSeq2Seq(nn.Module):
    """Transformer翻译模型: Encoder-Decoder结构"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, num_heads=8,
                 num_layers=3, d_ff=512, dropout=0.1, max_len=100,
                 pad_idx=3, bos_idx=0, eos_idx=1):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.max_len = max_len
        
        # embedding + 位置编码
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        
        # N层encoder和decoder
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.out_proj = nn.Linear(d_model, tgt_vocab_size)
        
        # xavier初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def make_src_mask(self, src):
        """padding mask"""
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
    
    def make_tgt_mask(self, tgt):
        """padding mask + causal mask (下三角)"""
        bs, tgt_len = tgt.size()
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).unsqueeze(0).unsqueeze(0)
        return pad_mask & (causal_mask.bool())
    
    def encode(self, src, src_mask):
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return x
    
    def decode(self, tgt, enc_out, tgt_mask, src_mask):
        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        for layer in self.dec_layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return x
    
    def forward(self, src, tgt):
        """训练: 并行计算所有位置"""
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, tgt_mask, src_mask)
        return F.log_softmax(self.out_proj(dec_out), dim=-1)
    
    def predict(self, src):
        """推理: 自回归逐词生成"""
        bs = src.size(0)
        device = src.device
        src_mask = self.make_src_mask(src)
        enc_out = self.encode(src, src_mask)
        
        # 从BOS开始
        tgt = torch.full((bs, 1), self.bos_idx, dtype=torch.long, device=device)
        for _ in range(self.max_len - 1):
            tgt_mask = self.make_tgt_mask(tgt)
            dec_out = self.decode(tgt, enc_out, tgt_mask, src_mask)
            next_tok = self.out_proj(dec_out[:, -1, :]).argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_tok], dim=1)
            if (next_tok == self.eos_idx).all():
                break
        return tgt
