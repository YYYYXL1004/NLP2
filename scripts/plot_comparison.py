"""
实验对比图 - 精简版
只生成2张图: 训练曲线 + 最终结果柱状图
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Linux兼容字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def load_logs():
    logs = {}
    for name in ['rnn', 'rnn_attn', 'lstm', 'lstm_attn', 'gru', 'gru_attn', 'transformer']:
        path = f"logs/{name}.json"
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                logs[name] = json.load(f)
    return logs


def plot_training_curves(logs):
    """2x2训练曲线图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 左上: 无Attention Loss
    ax = axes[0, 0]
    for name, color in [('rnn', 'blue'), ('lstm', 'green'), ('gru', 'orange')]:
        if name in logs:
            epochs = [x['epoch'] for x in logs[name]['logs']]
            losses = [x['loss'] for x in logs[name]['logs']]
            ax.plot(epochs, losses, color=color, label=name.upper(), linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss (without Attention)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 右上: 有Attention Loss
    ax = axes[0, 1]
    styles = {'rnn_attn': ('blue', 'RNN+Att'), 'lstm_attn': ('green', 'LSTM+Att'), 
              'gru_attn': ('orange', 'GRU+Att'), 'transformer': ('red', 'Transformer')}
    for name, (color, label) in styles.items():
        if name in logs:
            epochs = [x['epoch'] for x in logs[name]['logs']]
            losses = [x['loss'] for x in logs[name]['logs']]
            ax.plot(epochs, losses, color=color, label=label, linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss (with Attention)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 左下: 无Attention BLEU
    ax = axes[1, 0]
    for name, color in [('rnn', 'blue'), ('lstm', 'green'), ('gru', 'orange')]:
        if name in logs:
            epochs = [x['epoch'] for x in logs[name]['logs']]
            bleus = [x['bleu'] for x in logs[name]['logs']]
            ax.plot(epochs, bleus, color=color, label=name.upper(), linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('BLEU')
    ax.set_title('BLEU (without Attention)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 右下: 有Attention BLEU
    ax = axes[1, 1]
    for name, (color, label) in styles.items():
        if name in logs:
            epochs = [x['epoch'] for x in logs[name]['logs']]
            bleus = [x['bleu'] for x in logs[name]['logs']]
            ax.plot(epochs, bleus, color=color, label=label, linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('BLEU')
    ax.set_title('BLEU (with Attention)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/training_curves.png', dpi=150, bbox_inches='tight')
    print("Saved: figures/training_curves.png")
    plt.close()


def plot_final_results(logs):
    """最终结果柱状图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = ['RNN', 'LSTM', 'GRU', 'RNN+Att', 'LSTM+Att', 'GRU+Att', 'Transformer']
    keys = ['rnn', 'lstm', 'gru', 'rnn_attn', 'lstm_attn', 'gru_attn', 'transformer']
    colors = ['#6baed6', '#6baed6', '#6baed6', '#fd8d3c', '#fd8d3c', '#fd8d3c', '#e6550d']
    
    bleus = []
    for k in keys:
        if k in logs:
            bleus.append(logs[k]['logs'][-1]['bleu'])
        else:
            bleus.append(0)
    
    bars = ax.bar(models, bleus, color=colors, edgecolor='black', linewidth=1.2)
    
    # 添加数值标签
    for bar, bleu in zip(bars, bleus):
        ax.annotate(f'{bleu:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 添加分组标注
    ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=5.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(1, max(bleus)*0.95, 'No Attention', ha='center', fontsize=10, color='gray')
    ax.text(4, max(bleus)*0.95, 'With Attention', ha='center', fontsize=10, color='gray')
    ax.text(6, max(bleus)*0.95, 'Transformer', ha='center', fontsize=10, color='gray')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('BLEU Score', fontsize=12)
    ax.set_title('Final BLEU Score Comparison', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('figures/final_results.png', dpi=150, bbox_inches='tight')
    print("Saved: figures/final_results.png")
    plt.close()


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    logs = load_logs()
    print(f"Loaded {len(logs)} logs")
    
    plot_training_curves(logs)
    plot_final_results(logs)
    print("\nDone!")
