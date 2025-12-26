"""
生成实验对比图 - 用于报告
"""
import json
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_logs():
    """加载所有日志"""
    logs = {}
    log_dir = "logs"
    for name in ['rnn', 'rnn_attn', 'lstm', 'lstm_attn', 'gru', 'gru_attn', 'transformer']:
        path = os.path.join(log_dir, f"{name}.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                logs[name] = json.load(f)
    return logs

def plot_loss_comparison(logs):
    """Loss对比曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 无Attention组
    colors1 = {'rnn': 'blue', 'lstm': 'green', 'gru': 'orange'}
    for name, color in colors1.items():
        if name in logs:
            epochs = [x['epoch'] for x in logs[name]['logs']]
            losses = [x['loss'] for x in logs[name]['logs']]
            ax1.plot(epochs, losses, f'{color}', label=name.upper(), linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('RNN变体对比 (无Attention)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 有Attention组 + Transformer
    colors2 = {'rnn_attn': 'blue', 'lstm_attn': 'green', 'gru_attn': 'orange', 'transformer': 'red'}
    labels2 = {'rnn_attn': 'RNN+Att', 'lstm_attn': 'LSTM+Att', 'gru_attn': 'GRU+Att', 'transformer': 'Transformer'}
    for name, color in colors2.items():
        if name in logs:
            epochs = [x['epoch'] for x in logs[name]['logs']]
            losses = [x['loss'] for x in logs[name]['logs']]
            ax2.plot(epochs, losses, color, label=labels2[name], linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Attention机制对比', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/loss_comparison.png', dpi=150, bbox_inches='tight')
    print("保存: figures/loss_comparison.png")
    plt.close()

def plot_bleu_comparison(logs):
    """BLEU对比曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 无Attention组
    colors1 = {'rnn': 'blue', 'lstm': 'green', 'gru': 'orange'}
    for name, color in colors1.items():
        if name in logs:
            epochs = [x['epoch'] for x in logs[name]['logs']]
            bleus = [x['bleu'] for x in logs[name]['logs']]
            ax1.plot(epochs, bleus, color, label=name.upper(), linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('BLEU Score', fontsize=12)
    ax1.set_title('RNN变体对比 (无Attention)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 有Attention组 + Transformer
    colors2 = {'rnn_attn': 'blue', 'lstm_attn': 'green', 'gru_attn': 'orange', 'transformer': 'red'}
    labels2 = {'rnn_attn': 'RNN+Att', 'lstm_attn': 'LSTM+Att', 'gru_attn': 'GRU+Att', 'transformer': 'Transformer'}
    for name, color in colors2.items():
        if name in logs:
            epochs = [x['epoch'] for x in logs[name]['logs']]
            bleus = [x['bleu'] for x in logs[name]['logs']]
            ax2.plot(epochs, bleus, color, label=labels2[name], linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('BLEU Score', fontsize=12)
    ax2.set_title('Attention机制对比', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/bleu_comparison.png', dpi=150, bbox_inches='tight')
    print("保存: figures/bleu_comparison.png")
    plt.close()

def plot_attention_effect(logs):
    """Attention效果对比柱状图"""
    models = ['RNN', 'LSTM', 'GRU']
    without_att = []
    with_att = []
    
    for m in ['rnn', 'lstm', 'gru']:
        if m in logs:
            without_att.append(logs[m]['logs'][-1]['bleu'])
        if f'{m}_attn' in logs:
            with_att.append(logs[f'{m}_attn']['logs'][-1]['bleu'])
    
    x = range(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar([i - width/2 for i in x], without_att, width, label='无Attention', color='steelblue')
    bars2 = ax.bar([i + width/2 for i in x], with_att, width, label='有Attention', color='coral')
    
    # 添加Transformer
    if 'transformer' in logs:
        trans_bleu = logs['transformer']['logs'][-1]['bleu']
        ax.axhline(y=trans_bleu, color='red', linestyle='--', linewidth=2, label=f'Transformer ({trans_bleu:.1f})')
    
    ax.set_xlabel('模型', fontsize=12)
    ax.set_ylabel('BLEU Score', fontsize=12)
    ax.set_title('Attention机制效果对比', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar in bars1:
        ax.annotate(f'{bar.get_height():.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        ax.annotate(f'{bar.get_height():.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/attention_effect.png', dpi=150, bbox_inches='tight')
    print("保存: figures/attention_effect.png")
    plt.close()

def plot_final_summary(logs):
    """最终结果汇总图"""
    models = ['RNN', 'LSTM', 'GRU', 'RNN+Att', 'LSTM+Att', 'GRU+Att', 'Transformer']
    keys = ['rnn', 'lstm', 'gru', 'rnn_attn', 'lstm_attn', 'gru_attn', 'transformer']
    bleus = []
    colors = ['#4e79a7', '#4e79a7', '#4e79a7', '#f28e2b', '#f28e2b', '#f28e2b', '#e15759']
    
    for k in keys:
        if k in logs:
            bleus.append(logs[k]['logs'][-1]['bleu'])
        else:
            bleus.append(0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(models, bleus, color=colors, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('模型', fontsize=12)
    ax.set_ylabel('BLEU Score', fontsize=12)
    ax.set_title('所有模型BLEU分数对比', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, bleu in zip(bars, bleus):
        ax.annotate(f'{bleu:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figures/final_summary.png', dpi=150, bbox_inches='tight')
    print("保存: figures/final_summary.png")
    plt.close()

if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    logs = load_logs()
    print(f"加载了 {len(logs)} 个模型的日志")
    
    plot_loss_comparison(logs)
    plot_bleu_comparison(logs)
    plot_attention_effect(logs)
    plot_final_summary(logs)
    print("\n所有图表生成完成!")
