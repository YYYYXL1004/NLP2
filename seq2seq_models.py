"""
Seq2Seq模型 - 机器翻译
包含RNN/LSTM/GRU及Attention机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 复用的seq2seq-rnn里的逻辑
def load_data(num_train=-1):
    """加载中英翻译数据"""
    zh_sents = {}
    en_sents = {}
    for split in ['train', 'val', 'test']:
        zh_sents[split] = []
        en_sents[split] = []
        with open(f"data/zh_en_{split}.txt", encoding='utf-8') as f:
            for line in f.readlines():
                zh, en = line.strip().split("\t")
                zh_sents[split].append(zh.split())
                en_sents[split].append(en.split())
    
    if num_train != -1:
        zh_sents['train'] = zh_sents['train'][:num_train]
        en_sents['train'] = en_sents['train'][:num_train]
    
    print("训练集/验证集/测试集:", len(zh_sents['train']), len(zh_sents['val']), len(zh_sents['test']))
    return zh_sents, en_sents


class Vocab:
    """词表"""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        for w in ["[BOS]", "[EOS]", "[UNK]", "[PAD]"]:
            self.word2idx[w] = len(self.idx2word)
            self.idx2word.append(w)
    
    def add_sent(self, sent):
        for word in sent:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)
    
    def index(self, word):
        return self.word2idx.get(word, self.word2idx["[UNK]"])
    
    def encode(self, sent, max_len):
        ids = [self.word2idx["[BOS]"]] + [self.index(w) for w in sent][:max_len] + [self.word2idx["[EOS]"]]
        return ids
    
    def decode(self, ids, strip=False):
        words = []
        for i in ids:
            w = self.idx2word[i]
            if strip and w in ["[BOS]", "[EOS]", "[PAD]"]:
                continue
            words.append(w)
        return words
    
    def __len__(self):
        return len(self.idx2word)

class RNNCell(nn.Module):
    """h_new = tanh(W_ih * x + W_hh * h)"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.Wih = nn.Linear(input_size, hidden_size)
        self.Whh = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h):
        return torch.tanh(self.Wih(x) + self.Whh(h))


class LSTMCell(nn.Module):
    """LSTM: 输入门i, 遗忘门f, 候选g, 输出门o"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.Wi = nn.Linear(input_size, hidden_size)
        self.Wf = nn.Linear(input_size, hidden_size)
        self.Wg = nn.Linear(input_size, hidden_size)
        self.Wo = nn.Linear(input_size, hidden_size)
        self.Ui = nn.Linear(hidden_size, hidden_size)
        self.Uf = nn.Linear(hidden_size, hidden_size)
        self.Ug = nn.Linear(hidden_size, hidden_size)
        self.Uo = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, hc):
        h, c = hc
        i = torch.sigmoid(self.Wi(x) + self.Ui(h))  # 输入门
        f = torch.sigmoid(self.Wf(x) + self.Uf(h))  # 遗忘门
        g = torch.tanh(self.Wg(x) + self.Ug(h))     # 候选记忆
        o = torch.sigmoid(self.Wo(x) + self.Uo(h))  # 输出门
        c_new = f * c + i * g  # 更新记忆
        h_new = o * torch.tanh(c_new)  # 输出
        return h_new, c_new


class GRUCell(nn.Module):
    """GRU: 重置门r, 更新门z"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.Wr = nn.Linear(input_size, hidden_size)
        self.Wz = nn.Linear(input_size, hidden_size)
        self.Wn = nn.Linear(input_size, hidden_size)
        self.Ur = nn.Linear(hidden_size, hidden_size)
        self.Uz = nn.Linear(hidden_size, hidden_size)
        self.Un = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h):
        r = torch.sigmoid(self.Wr(x) + self.Ur(h))  # 重置门
        z = torch.sigmoid(self.Wz(x) + self.Uz(h))  # 更新门
        n = torch.tanh(self.Wn(x) + self.Un(r * h)) # 候选隐状态
        return (1 - z) * h + z * n  # 插值更新


# ========== Attention ==========
class Attention(nn.Module):
    """Bahdanau加性注意力"""
    def __init__(self, hidden_size):
        super().__init__()
        self.Wh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ws = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, dec_h, enc_hs):
        # dec_h: (batch, hidden), enc_hs: (batch, src_len, hidden)
        src_len = enc_hs.size(1)
        dec_h = dec_h.unsqueeze(1).expand(-1, src_len, -1)
        # 计算注意力分数
        energy = self.v(torch.tanh(self.Wh(enc_hs) + self.Ws(dec_h))).squeeze(-1)
        attn = F.softmax(energy, dim=-1)  # 归一化
        context = torch.bmm(attn.unsqueeze(1), enc_hs).squeeze(1)  # 加权求和
        return context, attn

# ========== Encoder ==========
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.rnn = RNNCell(emb_dim, hidden_size)
    
    def forward(self, x, h):
        return self.rnn(self.embed(x), h)

class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.lstm = LSTMCell(emb_dim, hidden_size)
    
    def forward(self, x, hc):
        return self.lstm(self.embed(x), hc)

class EncoderGRU(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.gru = GRUCell(emb_dim, hidden_size)
    
    def forward(self, x, h):
        return self.gru(self.embed(x), h)

# ========== Decoder ==========

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.rnn = RNNCell(emb_dim, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, h):
        h = self.rnn(self.embed(x), h)
        return F.log_softmax(self.out(h), dim=-1), h

class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.lstm = LSTMCell(emb_dim, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hc):
        h, c = self.lstm(self.embed(x), hc)
        return F.log_softmax(self.out(h), dim=-1), (h, c)

class DecoderGRU(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.gru = GRUCell(emb_dim, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, h):
        h = self.gru(self.embed(x), h)
        return F.log_softmax(self.out(h), dim=-1), h

# ========== Decoder with Attention ==========

class DecoderRNNAttn(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.attn = Attention(hidden_size)
        self.rnn = RNNCell(emb_dim + hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size * 2, vocab_size)
    
    def forward(self, x, h, enc_hs):
        emb = self.embed(x)
        ctx, attn_w = self.attn(h, enc_hs)
        h = self.rnn(torch.cat([emb, ctx], dim=-1), h)
        out = F.log_softmax(self.out(torch.cat([h, ctx], dim=-1)), dim=-1)
        return out, h, attn_w

class DecoderLSTMAttn(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.attn = Attention(hidden_size)
        self.lstm = LSTMCell(emb_dim + hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size * 2, vocab_size)
    
    def forward(self, x, hc, enc_hs):
        h, c = hc
        emb = self.embed(x)
        ctx, attn_w = self.attn(h, enc_hs)
        h, c = self.lstm(torch.cat([emb, ctx], dim=-1), (h, c))
        out = F.log_softmax(self.out(torch.cat([h, ctx], dim=-1)), dim=-1)
        return out, (h, c), attn_w

class DecoderGRUAttn(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.attn = Attention(hidden_size)
        self.gru = GRUCell(emb_dim + hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size * 2, vocab_size)
    
    def forward(self, x, h, enc_hs):
        emb = self.embed(x)
        ctx, attn_w = self.attn(h, enc_hs)
        h = self.gru(torch.cat([emb, ctx], dim=-1), h)
        out = F.log_softmax(self.out(torch.cat([h, ctx], dim=-1)), dim=-1)
        return out, h, attn_w


# ========== Seq2Seq Models ==========

class Seq2SeqRNN(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb_dim, hidden_size, max_len):
        super().__init__()
        self.tgt_vocab = tgt_vocab
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.encoder = EncoderRNN(len(src_vocab), emb_dim, hidden_size)
        self.decoder = DecoderRNN(len(tgt_vocab), emb_dim, hidden_size)
    
    def forward(self, src, tgt):
        """训练: teacher forcing"""
        bs, src_len = src.size()
        device = src.device
        h = torch.zeros(bs, self.hidden_size, device=device)
        
        # 编码: 逐词输入encoder
        enc_hs = []
        for i in range(src_len):
            h = self.encoder(src[:, i], h)
            enc_hs.append(h)
        enc_hs = torch.stack(enc_hs, dim=1)
        
        # 解码: 用真实目标作为输入
        outputs = []
        for i in range(tgt.size(1)):
            out, h = self.decoder(tgt[:, i], h)
            outputs.append(out)
        return torch.stack(outputs, dim=1)
    
    def predict(self, src):
        """推理: 自回归生成"""
        bs = src.size(0)
        device = src.device
        h = torch.zeros(bs, self.hidden_size, device=device)
        
        # 编码
        for i in range(src.size(1)):
            h = self.encoder(src[:, i], h)
        
        # 解码: 用上一步预测作为输入
        inp = torch.full((bs,), self.tgt_vocab.word2idx["[BOS]"], device=device, dtype=torch.long)
        preds = [inp]
        for _ in range(self.max_len):
            out, h = self.decoder(inp, h)
            inp = out.argmax(-1)  # 贪心选择
            preds.append(inp)
            if inp.item() == self.tgt_vocab.word2idx["[EOS]"]:
                break
        return torch.stack(preds, dim=-1)

class Seq2SeqLSTM(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb_dim, hidden_size, max_len):
        super().__init__()
        self.tgt_vocab = tgt_vocab
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.encoder = EncoderLSTM(len(src_vocab), emb_dim, hidden_size)
        self.decoder = DecoderLSTM(len(tgt_vocab), emb_dim, hidden_size)
    
    def forward(self, src, tgt):
        """训练: teacher forcing, LSTM需要维护h和c"""
        bs, src_len = src.size()
        device = src.device
        h = torch.zeros(bs, self.hidden_size, device=device)
        c = torch.zeros(bs, self.hidden_size, device=device)
        
        # 编码
        enc_hs = []
        for i in range(src_len):
            h, c = self.encoder(src[:, i], (h, c))
            enc_hs.append(h)
        enc_hs = torch.stack(enc_hs, dim=1)
        
        # 解码
        outputs = []
        for i in range(tgt.size(1)):
            out, (h, c) = self.decoder(tgt[:, i], (h, c))
            outputs.append(out)
        return torch.stack(outputs, dim=1)
    
    def predict(self, src):
        """推理: 自回归生成"""
        bs = src.size(0)
        device = src.device
        h = torch.zeros(bs, self.hidden_size, device=device)
        c = torch.zeros(bs, self.hidden_size, device=device)
        
        for i in range(src.size(1)):
            h, c = self.encoder(src[:, i], (h, c))
        
        inp = torch.full((bs,), self.tgt_vocab.word2idx["[BOS]"], device=device, dtype=torch.long)
        preds = [inp]
        for _ in range(self.max_len):
            out, (h, c) = self.decoder(inp, (h, c))
            inp = out.argmax(-1)
            preds.append(inp)
            if inp.item() == self.tgt_vocab.word2idx["[EOS]"]:
                break
        return torch.stack(preds, dim=-1)


class Seq2SeqGRU(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb_dim, hidden_size, max_len):
        super().__init__()
        self.tgt_vocab = tgt_vocab
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.encoder = EncoderGRU(len(src_vocab), emb_dim, hidden_size)
        self.decoder = DecoderGRU(len(tgt_vocab), emb_dim, hidden_size)
    
    def forward(self, src, tgt):
        """训练: teacher forcing"""
        bs, src_len = src.size()
        device = src.device
        h = torch.zeros(bs, self.hidden_size, device=device)
        
        # 编码
        enc_hs = []
        for i in range(src_len):
            h = self.encoder(src[:, i], h)
            enc_hs.append(h)
        enc_hs = torch.stack(enc_hs, dim=1)
        
        # 解码
        outputs = []
        for i in range(tgt.size(1)):
            out, h = self.decoder(tgt[:, i], h)
            outputs.append(out)
        return torch.stack(outputs, dim=1)
    
    def predict(self, src):
        """推理: 自回归生成"""
        bs = src.size(0)
        device = src.device
        h = torch.zeros(bs, self.hidden_size, device=device)
        
        for i in range(src.size(1)):
            h = self.encoder(src[:, i], h)
        
        inp = torch.full((bs,), self.tgt_vocab.word2idx["[BOS]"], device=device, dtype=torch.long)
        preds = [inp]
        for _ in range(self.max_len):
            out, h = self.decoder(inp, h)
            inp = out.argmax(-1)
            preds.append(inp)
            if inp.item() == self.tgt_vocab.word2idx["[EOS]"]:
                break
        return torch.stack(preds, dim=-1)

# ========== Seq2Seq with Attention ==========

class Seq2SeqRNNWithAttention(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb_dim, hidden_size, max_len):
        super().__init__()
        self.tgt_vocab = tgt_vocab
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.encoder = EncoderRNN(len(src_vocab), emb_dim, hidden_size)
        self.decoder = DecoderRNNAttn(len(tgt_vocab), emb_dim, hidden_size)
    
    def forward(self, src, tgt):
        """训练: 带attention的decoder每步都看encoder输出"""
        bs, src_len = src.size()
        device = src.device
        h = torch.zeros(bs, self.hidden_size, device=device)
        
        # 编码: 保存所有隐状态供attention使用
        enc_hs = []
        for i in range(src_len):
            h = self.encoder(src[:, i], h)
            enc_hs.append(h)
        enc_hs = torch.stack(enc_hs, dim=1)
        
        # 解码: 每步计算attention
        outputs = []
        for i in range(tgt.size(1)):
            out, h, _ = self.decoder(tgt[:, i], h, enc_hs)
            outputs.append(out)
        return torch.stack(outputs, dim=1)
    
    def predict(self, src):
        """推理: 自回归 + attention"""
        bs = src.size(0)
        device = src.device
        h = torch.zeros(bs, self.hidden_size, device=device)
        
        enc_hs = []
        for i in range(src.size(1)):
            h = self.encoder(src[:, i], h)
            enc_hs.append(h)
        enc_hs = torch.stack(enc_hs, dim=1)
        
        inp = torch.full((bs,), self.tgt_vocab.word2idx["[BOS]"], device=device, dtype=torch.long)
        preds = [inp]
        for _ in range(self.max_len):
            out, h, _ = self.decoder(inp, h, enc_hs)
            inp = out.argmax(-1)
            preds.append(inp)
            if inp.item() == self.tgt_vocab.word2idx["[EOS]"]:
                break
        return torch.stack(preds, dim=-1)

class Seq2SeqLSTMWithAttention(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb_dim, hidden_size, max_len):
        super().__init__()
        self.tgt_vocab = tgt_vocab
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.encoder = EncoderLSTM(len(src_vocab), emb_dim, hidden_size)
        self.decoder = DecoderLSTMAttn(len(tgt_vocab), emb_dim, hidden_size)
    
    def forward(self, src, tgt):
        """训练: LSTM + attention"""
        bs, src_len = src.size()
        device = src.device
        h = torch.zeros(bs, self.hidden_size, device=device)
        c = torch.zeros(bs, self.hidden_size, device=device)
        
        # 编码
        enc_hs = []
        for i in range(src_len):
            h, c = self.encoder(src[:, i], (h, c))
            enc_hs.append(h)
        enc_hs = torch.stack(enc_hs, dim=1)
        
        # 解码
        outputs = []
        for i in range(tgt.size(1)):
            out, (h, c), _ = self.decoder(tgt[:, i], (h, c), enc_hs)
            outputs.append(out)
        return torch.stack(outputs, dim=1)
    
    def predict(self, src):
        """推理: 自回归 + attention"""
        bs = src.size(0)
        device = src.device
        h = torch.zeros(bs, self.hidden_size, device=device)
        c = torch.zeros(bs, self.hidden_size, device=device)
        
        enc_hs = []
        for i in range(src.size(1)):
            h, c = self.encoder(src[:, i], (h, c))
            enc_hs.append(h)
        enc_hs = torch.stack(enc_hs, dim=1)
        
        inp = torch.full((bs,), self.tgt_vocab.word2idx["[BOS]"], device=device, dtype=torch.long)
        preds = [inp]
        for _ in range(self.max_len):
            out, (h, c), _ = self.decoder(inp, (h, c), enc_hs)
            inp = out.argmax(-1)
            preds.append(inp)
            if inp.item() == self.tgt_vocab.word2idx["[EOS]"]:
                break
        return torch.stack(preds, dim=-1)

class Seq2SeqGRUWithAttention(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb_dim, hidden_size, max_len):
        super().__init__()
        self.tgt_vocab = tgt_vocab
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.encoder = EncoderGRU(len(src_vocab), emb_dim, hidden_size)
        self.decoder = DecoderGRUAttn(len(tgt_vocab), emb_dim, hidden_size)
    
    def forward(self, src, tgt):
        """训练: GRU + attention"""
        bs, src_len = src.size()
        device = src.device
        h = torch.zeros(bs, self.hidden_size, device=device)
        
        # 编码
        enc_hs = []
        for i in range(src_len):
            h = self.encoder(src[:, i], h)
            enc_hs.append(h)
        enc_hs = torch.stack(enc_hs, dim=1)
        
        # 解码
        outputs = []
        for i in range(tgt.size(1)):
            out, h, _ = self.decoder(tgt[:, i], h, enc_hs)
            outputs.append(out)
        return torch.stack(outputs, dim=1)
    
    def predict(self, src):
        """推理: 自回归 + attention"""
        bs = src.size(0)
        device = src.device
        h = torch.zeros(bs, self.hidden_size, device=device)
        
        enc_hs = []
        for i in range(src.size(1)):
            h = self.encoder(src[:, i], h)
            enc_hs.append(h)
        enc_hs = torch.stack(enc_hs, dim=1)
        
        inp = torch.full((bs,), self.tgt_vocab.word2idx["[BOS]"], device=device, dtype=torch.long)
        preds = [inp]
        for _ in range(self.max_len):
            out, h, _ = self.decoder(inp, h, enc_hs)
            inp = out.argmax(-1)
            preds.append(inp)
            if inp.item() == self.tgt_vocab.word2idx["[EOS]"]:
                break
        return torch.stack(preds, dim=-1)

# ========== DataLoader ==========
def padding(ids, max_len, pad_id):
    max_len += 2  # BOS和EOS
    result = np.ones(max_len, dtype=np.int32) * pad_id
    length = min(len(ids), max_len)
    result[:length] = ids[:length]
    return result

def collate(batch):
    src = torch.stack([torch.LongTensor(x[0]) for x in batch])
    tgt = torch.stack([torch.LongTensor(x[1]) for x in batch])
    return src, tgt

def create_dataloader(zh_sents, en_sents, zh_vocab, en_vocab, max_len, batch_size, pad_id):
    loaders = {}
    for split in ['train', 'val', 'test']:
        data = []
        for zh, en in zip(zh_sents[split], en_sents[split]):
            src = padding(zh_vocab.encode(zh, max_len), max_len, pad_id)
            tgt = padding(en_vocab.encode(en, max_len), max_len, pad_id)
            data.append((src, tgt))
        shuffle = (split == 'train')
        loaders[split] = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
    return loaders['train'], loaders['val'], loaders['test']
