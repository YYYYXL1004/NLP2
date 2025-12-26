"""
消融实验可视化
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_logs(log_dir='logs'):
    """加载所有日志"""
    logs = {}
    for f in os.listdir(log_dir):
        if f.endswith('.json'):
            name = f.replace('.json', '')
            with open(os.path.join(log_dir, f), 'r', encoding='utf-8') as fp:
                logs[name] = json.load(fp)
    return logs


def get_test_bleu_from_log(name, log_dir='logs'):
    """从log文件中提取测试集BLEU"""
    log_path = os.path.join(log_dir, f"{name}.log")
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if 'Test BLEU:' in line:
                    return float(line.split('Test BLEU:')[1].strip())
    return 0


def get_best_bleu(log_data):
    """获取最佳验证BLEU"""
    return max(x['bleu'] for x in log_data['logs'])


def plot_ablation_hidden_size(logs, save_dir='figures'):
    """消融实验1: Hidden Size对LSTM+Att的影响"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    configs = [
        ('lstm_attn_h128', '128', '#74c476'),
        ('lstm_attn_h256', '256', '#31a354'),
        ('lstm_attn_h512', '512', '#006d2c'),
    ]
    
    # 训练曲线
    for key, label, color in configs:
        if key in logs:
            epochs = [x['epoch'] for x in logs[key]['logs']]
            bleus = [x['bleu'] for x in logs[key]['logs']]
            ax1.plot(epochs, bleus, '-o', label=f'hidden={label}', color=color, markersize=4)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation BLEU')
    ax1.set_title('LSTM+Attention: Hidden Size Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 柱状图 - 使用测试集BLEU
    sizes = ['128', '256', '512']
    keys = ['lstm_attn_h128', 'lstm_attn_h256', 'lstm_attn_h512']
    bleus = [get_test_bleu_from_log(k) for k in keys]
    colors = ['#74c476', '#31a354', '#006d2c']
    
    bars = ax2.bar(sizes, bleus, color=colors, edgecolor='black', linewidth=1.2)
    for bar, bleu in zip(bars, bleus):
        ax2.annotate(f'{bleu:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_xlabel('Hidden Size')
    ax2.set_ylabel('Test BLEU')
    ax2.set_title('LSTM+Attention: Test BLEU by Hidden Size')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_hidden_size.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir}/ablation_hidden_size.png")
    plt.close()


def plot_ablation_transformer_layers(logs, save_dir='figures'):
    """消融实验2: Transformer层数的影响"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    configs = [
        ('transformer_L1', 'L=1', '#9ecae1'),
        ('transformer_L2', 'L=2', '#4292c6'),
        ('transformer_L3', 'L=3', '#2171b5'),
        ('transformer_L4', 'L=4', '#084594'),
    ]
    
    # 训练曲线
    for key, label, color in configs:
        if key in logs:
            epochs = [x['epoch'] for x in logs[key]['logs']]
            bleus = [x['bleu'] for x in logs[key]['logs']]
            ax1.plot(epochs, bleus, '-o', label=label, color=color, markersize=4)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation BLEU')
    ax1.set_title('Transformer: Number of Layers Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 柱状图 - 使用测试集BLEU
    layers = ['1', '2', '3', '4']
    keys = ['transformer_L1', 'transformer_L2', 'transformer_L3', 'transformer_L4']
    bleus = [get_test_bleu_from_log(k) for k in keys]
    colors = ['#9ecae1', '#4292c6', '#2171b5', '#084594']
    
    bars = ax2.bar(layers, bleus, color=colors, edgecolor='black', linewidth=1.2)
    for bar, bleu in zip(bars, bleus):
        ax2.annotate(f'{bleu:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_xlabel('Number of Layers')
    ax2.set_ylabel('Test BLEU')
    ax2.set_title('Transformer: Test BLEU by Layer Count')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_transformer_layers.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir}/ablation_transformer_layers.png")
    plt.close()


def plot_ablation_attention_heads(logs, save_dir='figures'):
    """消融实验3: Transformer注意力头数的影响"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    configs = [
        ('transformer_H1', 'H=1', '#fdae6b'),
        ('transformer_H4', 'H=4', '#f16913'),
        ('transformer_H8', 'H=8', '#d94801'),
    ]
    
    # 训练曲线
    for key, label, color in configs:
        if key in logs:
            epochs = [x['epoch'] for x in logs[key]['logs']]
            bleus = [x['bleu'] for x in logs[key]['logs']]
            ax1.plot(epochs, bleus, '-o', label=label, color=color, markersize=4)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation BLEU')
    ax1.set_title('Transformer: Number of Attention Heads Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 柱状图 - 使用测试集BLEU
    heads = ['1', '4', '8']
    keys = ['transformer_H1', 'transformer_H4', 'transformer_H8']
    bleus = [get_test_bleu_from_log(k) for k in keys]
    colors = ['#fdae6b', '#f16913', '#d94801']
    
    bars = ax2.bar(heads, bleus, color=colors, edgecolor='black', linewidth=1.2)
    for bar, bleu in zip(bars, bleus):
        ax2.annotate(f'{bleu:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_xlabel('Number of Attention Heads')
    ax2.set_ylabel('Test BLEU')
    ax2.set_title('Transformer: Test BLEU by Head Count')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_attention_heads.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir}/ablation_attention_heads.png")
    plt.close()


def plot_ablation_summary(logs, save_dir='figures'):
    """消融实验汇总图 - 使用测试集BLEU"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # 1. Hidden Size
    ax = axes[0]
    sizes = ['128', '256', '512']
    keys = ['lstm_attn_h128', 'lstm_attn_h256', 'lstm_attn_h512']
    bleus = [get_test_bleu_from_log(k) for k in keys]
    colors = ['#74c476', '#31a354', '#006d2c']
    bars = ax.bar(sizes, bleus, color=colors, edgecolor='black', linewidth=1.2)
    for bar, bleu in zip(bars, bleus):
        ax.annotate(f'{bleu:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xlabel('Hidden Size')
    ax.set_ylabel('Test BLEU')
    ax.set_title('(a) LSTM+Att: Hidden Size')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 30)
    
    # 2. Transformer Layers
    ax = axes[1]
    layers = ['1', '2', '3', '4']
    keys = ['transformer_L1', 'transformer_L2', 'transformer_L3', 'transformer_L4']
    bleus = [get_test_bleu_from_log(k) for k in keys]
    colors = ['#9ecae1', '#4292c6', '#2171b5', '#084594']
    bars = ax.bar(layers, bleus, color=colors, edgecolor='black', linewidth=1.2)
    for bar, bleu in zip(bars, bleus):
        ax.annotate(f'{bleu:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xlabel('Number of Layers')
    ax.set_ylabel('Test BLEU')
    ax.set_title('(b) Transformer: Layers')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 30)
    
    # 3. Attention Heads
    ax = axes[2]
    heads = ['1', '4', '8']
    keys = ['transformer_H1', 'transformer_H4', 'transformer_H8']
    bleus = [get_test_bleu_from_log(k) for k in keys]
    colors = ['#fdae6b', '#f16913', '#d94801']
    bars = ax.bar(heads, bleus, color=colors, edgecolor='black', linewidth=1.2)
    for bar, bleu in zip(bars, bleus):
        ax.annotate(f'{bleu:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xlabel('Number of Heads')
    ax.set_ylabel('Test BLEU')
    ax.set_title('(c) Transformer: Attention Heads')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 30)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_summary.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir}/ablation_summary.png")
    plt.close()


if __name__ == '__main__':
    logs = load_logs('logs')
    
    print("=== 消融实验可视化 ===")
    plot_ablation_hidden_size(logs)
    plot_ablation_transformer_layers(logs)
    plot_ablation_attention_heads(logs)
    plot_ablation_summary(logs)
    print("Done!")
