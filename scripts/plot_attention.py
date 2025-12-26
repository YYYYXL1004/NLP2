"""
Attention热力图可视化 - 精简版
只生成一张3x1对比图
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seq2seq_models import (
    Vocab, load_data,
    Seq2SeqRNNWithAttention, Seq2SeqLSTMWithAttention, Seq2SeqGRUWithAttention
)

# Linux兼容字体设置
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def get_attention_weights(model, src_sent, zh_vocab, en_vocab, device, max_len=10):
    """提取attention权重"""
    model.eval()
    src_words = src_sent.split()
    src_ids = zh_vocab.encode(src_words, max_len)
    src_tensor = torch.LongTensor([src_ids]).to(device)
    
    bs = 1
    h = torch.zeros(bs, model.hidden_size, device=device)
    
    # 编码
    if hasattr(model.encoder, 'lstm'):
        c = torch.zeros(bs, model.hidden_size, device=device)
        enc_hs = []
        for i in range(src_tensor.size(1)):
            h, c = model.encoder(src_tensor[:, i], (h, c))
            enc_hs.append(h)
        enc_hs = torch.stack(enc_hs, dim=1)
        state = (h, c)
    else:
        enc_hs = []
        for i in range(src_tensor.size(1)):
            h = model.encoder(src_tensor[:, i], h)
            enc_hs.append(h)
        enc_hs = torch.stack(enc_hs, dim=1)
        state = h
    
    # 解码
    inp = torch.full((bs,), en_vocab.word2idx["[BOS]"], device=device, dtype=torch.long)
    pred_words, attention_weights = [], []
    
    for _ in range(max_len + 2):
        with torch.no_grad():
            if hasattr(model.encoder, 'lstm'):
                out, state, attn_w = model.decoder(inp, state, enc_hs)
            else:
                out, state, attn_w = model.decoder(inp, state, enc_hs)
        
        attention_weights.append(attn_w.squeeze(0).cpu().numpy())
        inp = out.argmax(-1)
        word = en_vocab.idx2word[inp.item()]
        if word == '[EOS]':
            break
        pred_words.append(word)
    
    src_display = ['BOS'] + src_words[:max_len] + ['EOS']
    while len(src_display) < len(attention_weights[0]):
        src_display.append('PAD')
    
    return src_display, pred_words, np.array(attention_weights)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 加载词表
    zh_sents, en_sents = load_data(-1)
    zh_vocab, en_vocab = Vocab(), Vocab()
    for zh, en in zip(zh_sents['train'], en_sents['train']):
        zh_vocab.add_sent(zh)
        en_vocab.add_sent(en)
    
    os.makedirs('figures', exist_ok=True)
    
    # 只用一个典型句子
    test_sent = "我 爱 你"
    print(f"Source: {test_sent}")
    
    models_config = [
        ('RNN+Att', 'rnn_attn', Seq2SeqRNNWithAttention),
        ('LSTM+Att', 'lstm_attn', Seq2SeqLSTMWithAttention),
        ('GRU+Att', 'gru_attn', Seq2SeqGRUWithAttention),
    ]
    
    # 收集结果
    results = []
    for name, ckpt_key, model_cls in models_config:
        ckpt_path = f'checkpoints/{ckpt_key}_best.pt'
        if not os.path.exists(ckpt_path):
            print(f"Skip {name}: {ckpt_path} not found")
            continue
        
        model = model_cls(zh_vocab, en_vocab, 256, 256, 12)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        
        src_words, tgt_words, attn = get_attention_weights(model, test_sent, zh_vocab, en_vocab, device)
        print(f"{name}: {' '.join(tgt_words)}")
        results.append((name, src_words, tgt_words, attn))
    
    if not results:
        print("No models found!")
        return
    
    # 生成一张对比图
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1:
        axes = [axes]
    
    for ax, (name, src_words, tgt_words, attn) in zip(axes, results):
        attn = attn[:len(tgt_words), :len(src_words)]
        im = ax.imshow(attn, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(src_words)))
        ax.set_xticklabels(src_words, rotation=45, ha='right', fontsize=10)
        ax.set_yticks(range(len(tgt_words)))
        ax.set_yticklabels(tgt_words, fontsize=10)
        ax.set_xlabel('Source (Chinese)')
        ax.set_ylabel('Target (English)')
        ax.set_title(f'{name}\nOutput: {" ".join(tgt_words)}', fontsize=11)
        
        # 显示数值
        for i in range(len(tgt_words)):
            for j in range(len(src_words)):
                if attn[i, j] > 0.15:
                    color = 'white' if attn[i, j] > 0.5 else 'black'
                    ax.text(j, i, f'{attn[i, j]:.2f}', ha='center', va='center', color=color, fontsize=8)
    
    fig.colorbar(im, ax=axes, shrink=0.6, label='Attention Weight')
    plt.suptitle(f'Attention Heatmap Comparison\nSource: "{test_sent}"', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/attention_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: figures/attention_heatmap.png")
    plt.close()


if __name__ == '__main__':
    main()
