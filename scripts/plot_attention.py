"""
Attention热力图可视化
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import sys
import os
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seq2seq_models import (
    Vocab, load_data,
    Seq2SeqRNNWithAttention, Seq2SeqLSTMWithAttention, Seq2SeqGRUWithAttention
)


def get_chinese_font():
    """获取中文字体，如果没有则下载"""
    font_dir = os.path.join(os.path.dirname(__file__), 'fonts')
    font_path = os.path.join(font_dir, 'SimHei.ttf')
    
    if not os.path.exists(font_path):
        os.makedirs(font_dir, exist_ok=True)
        print("Downloading Chinese font (SimHei.ttf)...")
        url = "https://raw.githubusercontent.com/StellarCN/scp_zh/master/fonts/SimHei.ttf"
        try:
            urllib.request.urlretrieve(url, font_path)
            print(f"Font saved to {font_path}")
        except Exception as e:
            print(f"Failed to download font: {e}")
            print("Using default font (Chinese may not display correctly)")
            return None
    
    return FontProperties(fname=font_path)


def get_attention_weights(model, src_sent, zh_vocab, en_vocab, device, max_len=10):
    """提取attention权重"""
    model.eval()
    src_words = src_sent.split()
    src_ids = zh_vocab.encode(src_words, max_len)
    src_tensor = torch.LongTensor([src_ids]).to(device)
    
    bs = 1
    h = torch.zeros(bs, model.hidden_size, device=device)
    
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
    
    inp = torch.full((bs,), en_vocab.word2idx["[BOS]"], device=device, dtype=torch.long)
    pred_words, attention_weights = [], []
    
    max_output = 8
    for _ in range(max_output):
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
    
    # 源词标注
    src_display = ['[BOS]'] + src_words[:max_len] + ['[EOS]']
    
    return src_display, pred_words, np.array(attention_weights)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 获取中文字体
    chinese_font = get_chinese_font()
    
    zh_sents, en_sents = load_data(-1)
    zh_vocab, en_vocab = Vocab(), Vocab()
    for zh, en in zip(zh_sents['train'], en_sents['train']):
        zh_vocab.add_sent(zh)
        en_vocab.add_sent(en)
    
    os.makedirs('figures', exist_ok=True)
    
    test_sent = "我 爱 你"
    print(f"Source: {test_sent}")
    
    models_config = [
        ('RNN+Att', 'rnn_attn', Seq2SeqRNNWithAttention),
        ('LSTM+Att', 'lstm_attn', Seq2SeqLSTMWithAttention),
        ('GRU+Att', 'gru_attn', Seq2SeqGRUWithAttention),
    ]
    
    results = []
    for name, ckpt_key, model_cls in models_config:
        ckpt_path = f'checkpoints/{ckpt_key}_best.pt'
        if not os.path.exists(ckpt_path):
            print(f"Skip {name}: {ckpt_path} not found")
            continue
        
        model = model_cls(zh_vocab, en_vocab, 256, 256, 12)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model = model.to(device)  # 确保模型在正确设备上
        
        src_display, tgt_words, attn = get_attention_weights(
            model, test_sent, zh_vocab, en_vocab, device
        )
        output = ' '.join(tgt_words[:6])
        print(f"{name}: {output}")
        results.append((name, src_display, tgt_words[:6], attn, output))
    
    if not results:
        print("No models found!")
        return
    
    # 生成对比图 - 使用 gridspec 精确控制布局
    n = len(results)
    fig = plt.figure(figsize=(5*n + 1, 5))
    
    # 创建网格：n个子图 + 1个colorbar
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1]*n + [0.05], wspace=0.3)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    cax = fig.add_subplot(gs[0, n])  # colorbar 专用轴
    
    for ax, (name, src_display, tgt_words, attn, output) in zip(axes, results):
        tgt_len = len(tgt_words)
        src_len = min(len(src_display), attn.shape[1])
        attn_plot = attn[:tgt_len, :src_len]
        
        im = ax.imshow(attn_plot, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(range(src_len))
        # 使用中文字体显示源词
        if chinese_font:
            ax.set_xticklabels(src_display[:src_len], fontproperties=chinese_font, fontsize=11)
        else:
            ax.set_xticklabels(src_display[:src_len], fontsize=11)
        
        ax.set_yticks(range(tgt_len))
        ax.set_yticklabels(tgt_words, fontsize=10)
        ax.set_xlabel('Source (Chinese)', fontsize=10)
        ax.set_ylabel('Target (English)', fontsize=10)
        
        title_output = output if len(output) < 25 else output[:22] + '...'
        ax.set_title(f'{name}\n"{title_output}"', fontsize=11)
        
        for i in range(tgt_len):
            for j in range(src_len):
                if attn_plot[i, j] > 0.15:
                    color = 'white' if attn_plot[i, j] > 0.5 else 'black'
                    ax.text(j, i, f'{attn_plot[i, j]:.2f}', ha='center', va='center', 
                           color=color, fontsize=8)
    
    # colorbar 放在专用轴上
    fig.colorbar(im, cax=cax, label='Attention Weight')
    
    # 标题
    if chinese_font:
        fig.suptitle(f'Attention Heatmap: "{test_sent}"', fontproperties=chinese_font, fontsize=13, y=1.02)
    else:
        fig.suptitle(f'Attention Heatmap', fontsize=13, y=1.02)
    
    plt.savefig('figures/attention_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: figures/attention_heatmap.png")
    plt.close()


if __name__ == '__main__':
    main()
