"""
Attention热力图可视化
展示模型学到的源语言-目标语言对齐关系
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seq2seq_models import (
    Vocab, load_data,
    Seq2SeqRNNWithAttention, Seq2SeqLSTMWithAttention, Seq2SeqGRUWithAttention
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class AttentionExtractor:
    """提取Attention权重的包装器"""
    def __init__(self, model, zh_vocab, en_vocab, device):
        self.model = model.to(device).eval()
        self.zh_vocab = zh_vocab
        self.en_vocab = en_vocab
        self.device = device
    
    def translate_with_attention(self, src_sent, max_len=10):
        """翻译并返回attention权重"""
        # 编码源句
        src_words = src_sent.split()
        src_ids = self.zh_vocab.encode(src_words, max_len)
        src_tensor = torch.LongTensor([src_ids]).to(self.device)
        
        # 编码阶段
        bs = 1
        h = torch.zeros(bs, self.model.hidden_size, device=self.device)
        
        # 根据模型类型处理
        if hasattr(self.model.encoder, 'lstm'):  # LSTM
            c = torch.zeros(bs, self.model.hidden_size, device=self.device)
            enc_hs = []
            for i in range(src_tensor.size(1)):
                h, c = self.model.encoder(src_tensor[:, i], (h, c))
                enc_hs.append(h)
            enc_hs = torch.stack(enc_hs, dim=1)
            state = (h, c)
        else:  # RNN/GRU
            enc_hs = []
            for i in range(src_tensor.size(1)):
                h = self.model.encoder(src_tensor[:, i], h)
                enc_hs.append(h)
            enc_hs = torch.stack(enc_hs, dim=1)
            state = h
        
        # 解码阶段，收集attention权重
        inp = torch.full((bs,), self.en_vocab.word2idx["[BOS]"], device=self.device, dtype=torch.long)
        pred_words = []
        attention_weights = []
        
        for _ in range(max_len + 2):
            with torch.no_grad():
                if hasattr(self.model.encoder, 'lstm'):  # LSTM
                    out, state, attn_w = self.model.decoder(inp, state, enc_hs)
                else:  # RNN/GRU
                    out, state, attn_w = self.model.decoder(inp, state, enc_hs)
            
            attention_weights.append(attn_w.squeeze(0).cpu().numpy())
            inp = out.argmax(-1)
            word = self.en_vocab.idx2word[inp.item()]
            
            if word == '[EOS]':
                break
            pred_words.append(word)
        
        # 构建源词列表（包含BOS/EOS）
        src_display = ['[BOS]'] + src_words[:max_len] + ['[EOS]']
        # 补齐到实际长度
        while len(src_display) < len(attention_weights[0]):
            src_display.append('[PAD]')
        
        return src_display, pred_words, np.array(attention_weights)


def plot_attention_heatmap(src_words, tgt_words, attention, save_path, title="Attention"):
    """绘制attention热力图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 只取有效部分
    attn = attention[:len(tgt_words), :len(src_words)]
    
    im = ax.imshow(attn, cmap='Blues', aspect='auto')
    
    # 设置坐标轴
    ax.set_xticks(range(len(src_words)))
    ax.set_xticklabels(src_words, rotation=45, ha='right', fontsize=11)
    ax.set_yticks(range(len(tgt_words)))
    ax.set_yticklabels(tgt_words, fontsize=11)
    
    ax.set_xlabel('源语言 (中文)', fontsize=12)
    ax.set_ylabel('目标语言 (英文)', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # 添加颜色条
    plt.colorbar(im, ax=ax, label='Attention Weight')
    
    # 在格子中显示数值
    for i in range(len(tgt_words)):
        for j in range(len(src_words)):
            if attn[i, j] > 0.1:  # 只显示较大的值
                ax.text(j, i, f'{attn[i, j]:.2f}', ha='center', va='center', 
                       color='white' if attn[i, j] > 0.5 else 'black', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"保存: {save_path}")
    plt.close()


def plot_multi_attention(results, save_path):
    """多个模型的attention对比"""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
    if n == 1:
        axes = [axes]
    
    for ax, (name, src_words, tgt_words, attention) in zip(axes, results):
        attn = attention[:len(tgt_words), :len(src_words)]
        im = ax.imshow(attn, cmap='Blues', aspect='auto')
        
        ax.set_xticks(range(len(src_words)))
        ax.set_xticklabels(src_words, rotation=45, ha='right', fontsize=10)
        ax.set_yticks(range(len(tgt_words)))
        ax.set_yticklabels(tgt_words, fontsize=10)
        ax.set_title(name, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"保存: {save_path}")
    plt.close()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # 加载词表
    zh_sents, en_sents = load_data(-1)
    zh_vocab, en_vocab = Vocab(), Vocab()
    for zh, en in zip(zh_sents['train'], en_sents['train']):
        zh_vocab.add_sent(zh)
        en_vocab.add_sent(en)
    
    os.makedirs('figures', exist_ok=True)
    
    # 测试句子
    test_sentences = [
        "我 爱 你",
        "他 是 我 的 朋友",
        "今天 天气 很 好",
    ]
    
    # 加载带Attention的模型
    models_config = [
        ('RNN+Att', 'rnn', Seq2SeqRNNWithAttention),
        ('LSTM+Att', 'lstm', Seq2SeqLSTMWithAttention),
        ('GRU+Att', 'gru', Seq2SeqGRUWithAttention),
    ]
    
    for sent_idx, src_sent in enumerate(test_sentences):
        print(f"\n源句: {src_sent}")
        results = []
        
        for name, model_key, model_cls in models_config:
            ckpt_path = f'checkpoints/{model_key}_attn_best.pt'
            if not os.path.exists(ckpt_path):
                print(f"  跳过 {name}: 未找到 {ckpt_path}")
                continue
            
            # 加载模型
            model = model_cls(zh_vocab, en_vocab, 256, 256, 12)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            
            extractor = AttentionExtractor(model, zh_vocab, en_vocab, device)
            src_words, tgt_words, attention = extractor.translate_with_attention(src_sent)
            
            print(f"  {name}: {' '.join(tgt_words)}")
            
            # 单独保存
            plot_attention_heatmap(
                src_words, tgt_words, attention,
                f'figures/attention_{model_key}_{sent_idx}.png',
                f'{name} Attention'
            )
            
            results.append((name, src_words, tgt_words, attention))
        
        # 对比图
        if len(results) > 1:
            plot_multi_attention(results, f'figures/attention_compare_{sent_idx}.png')
    
    print("\nAttention热力图生成完成!")


if __name__ == '__main__':
    main()
