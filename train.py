"""
Seq2Seq Translation Training Script

整合所有模型（RNN、LSTM、GRU、Transformer）的主训练脚本。
支持通过命令行参数选择模型类型和是否使用attention。

Requirements: 7.1, 7.2
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sacrebleu.metrics import BLEU

from seq2seq_models import (
    Vocab, load_data, padding, collate, create_dataloader,
    Seq2SeqRNN, Seq2SeqLSTM, Seq2SeqGRU,
    Seq2SeqRNNWithAttention, Seq2SeqLSTMWithAttention, Seq2SeqGRUWithAttention
)
from transformer import TransformerSeq2Seq
from utils import TrainingLogger, Visualizer


def get_model(model_type: str, use_attention: bool, 
              src_vocab: Vocab, tgt_vocab: Vocab,
              embedding_dim: int, hidden_size: int, max_len: int,
              num_layers: int = 3, num_heads: int = 8, d_ff: int = 512,
              dropout: float = 0.1):
    """
    根据参数创建对应的模型
    
    Args:
        model_type: 模型类型 (rnn/lstm/gru/transformer)
        use_attention: 是否使用attention（仅对RNN变体有效）
        src_vocab: 源语言词表
        tgt_vocab: 目标语言词表
        embedding_dim: embedding维度
        hidden_size: 隐藏层维度
        max_len: 最大序列长度
        num_layers: Transformer层数
        num_heads: 注意力头数
        d_ff: FFN隐藏层维度
        dropout: dropout概率
    
    Returns:
        model: 创建的模型实例
    """
    if model_type == 'transformer':
        model = TransformerSeq2Seq(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len + 2,  # +2 for BOS and EOS
            pad_idx=src_vocab.word2idx['[PAD]'],
            bos_idx=src_vocab.word2idx['[BOS]'],
            eos_idx=src_vocab.word2idx['[EOS]']
        )
    elif model_type == 'lstm':
        if use_attention:
            model = Seq2SeqLSTMWithAttention(
                src_vocab, tgt_vocab, embedding_dim, hidden_size, max_len + 2
            )
        else:
            model = Seq2SeqLSTM(
                src_vocab, tgt_vocab, embedding_dim, hidden_size, max_len + 2
            )
    elif model_type == 'gru':
        if use_attention:
            model = Seq2SeqGRUWithAttention(
                src_vocab, tgt_vocab, embedding_dim, hidden_size, max_len + 2
            )
        else:
            model = Seq2SeqGRU(
                src_vocab, tgt_vocab, embedding_dim, hidden_size, max_len + 2
            )
    else:  # rnn
        if use_attention:
            model = Seq2SeqRNNWithAttention(
                src_vocab, tgt_vocab, embedding_dim, hidden_size, max_len + 2
            )
        else:
            model = Seq2SeqRNN(
                src_vocab, tgt_vocab, embedding_dim, hidden_size, max_len + 2
            )
    
    return model


def train_loop(model, optimizer, criterion, loader, device, model_type: str):
    """
    训练一个epoch
    
    Args:
        model: 模型
        optimizer: 优化器
        criterion: 损失函数
        loader: 数据加载器
        device: 设备
        model_type: 模型类型
    
    Returns:
        epoch_loss: 平均训练损失
    """
    model.train()
    epoch_loss = 0.0
    
    for src, tgt in tqdm(loader, desc="Training"):
        src = src.to(device)
        tgt = tgt.to(device)
        
        outputs = model(src, tgt)
        
        # 计算损失：预测outputs[:,:-1,:]与目标tgt[:,1:]对齐
        loss = criterion(
            outputs[:, :-1, :].reshape(-1, outputs.shape[-1]), 
            tgt[:, 1:].reshape(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # 梯度裁剪
        optimizer.step()
        
        epoch_loss += loss.item()
    
    epoch_loss /= len(loader)
    return epoch_loss



def test_loop(model, loader, tgt_vocab: Vocab, device, model_type: str):
    """
    在验证/测试集上评估模型
    
    Args:
        model: 模型
        loader: 数据加载器
        tgt_vocab: 目标语言词表
        device: 设备
        model_type: 模型类型
    
    Returns:
        hypotheses: 预测结果列表
        references: 参考答案列表
        bleu_score: BLEU分数
    """
    model.eval()
    bleu = BLEU(force=True)
    hypotheses, references = [], []
    
    for src, tgt in tqdm(loader, desc="Evaluating"):
        batch_size = len(src)
        for i in range(batch_size):
            _src = src[i].unsqueeze(0).to(device)  # 1 * L
            
            with torch.no_grad():
                outputs = model.predict(_src)  # 1 * L
            
            # 解码预测结果和参考答案
            ref = " ".join(tgt_vocab.decode(tgt[i].tolist(), strip_bos_eos_pad=True))
            hypo = " ".join(tgt_vocab.decode(outputs[0].cpu().tolist(), strip_bos_eos_pad=True))
            
            references.append(ref)
            hypotheses.append(hypo)
    
    score = bleu.corpus_score(hypotheses, [references]).score
    return hypotheses, references, score


def save_checkpoint(model, optimizer, epoch, best_bleu, filepath):
    """
    保存模型checkpoint
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        best_bleu: 最佳BLEU分数
        filepath: 保存路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_bleu': best_bleu
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """
    加载模型checkpoint
    
    Args:
        model: 模型
        optimizer: 优化器
        filepath: checkpoint路径
        device: 设备
    
    Returns:
        start_epoch: 起始epoch
        best_bleu: 最佳BLEU分数
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_bleu = checkpoint['best_bleu']
    print(f"Checkpoint loaded from {filepath}, resuming from epoch {start_epoch}")
    return start_epoch, best_bleu


def main():
    parser = argparse.ArgumentParser(description='Seq2Seq Translation Training')
    
    # 模型选择参数
    parser.add_argument('--model_type', type=str, default='lstm',
                        choices=['rnn', 'lstm', 'gru', 'transformer'],
                        help='模型类型: rnn/lstm/gru/transformer')
    parser.add_argument('--use_attention', action='store_true',
                        help='是否使用attention（仅对RNN变体有效）')
    
    # 数据参数
    parser.add_argument('--num_train', type=int, default=-1,
                        help='训练集大小，-1表示使用全部数据')
    parser.add_argument('--max_len', type=int, default=10,
                        help='句子最大长度')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批次大小')
    
    # 模型参数
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='embedding维度')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Transformer层数')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='注意力头数')
    parser.add_argument('--d_ff', type=int, default=512,
                        help='FFN隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout概率')
    
    # 训练参数
    parser.add_argument('--num_epoch', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='学习率')
    parser.add_argument('--optim', type=str, default='adam',
                        choices=['sgd', 'adam'],
                        help='优化器: sgd/adam')
    
    # 保存和恢复参数
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='checkpoint保存目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='从checkpoint恢复训练的路径')
    
    # 日志和可视化参数
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='日志保存目录')
    parser.add_argument('--fig_dir', type=str, default='figures',
                        help='图表保存目录')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='实验名称')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=1,
                        help='随机种子')
    parser.add_argument('--no_visualize', action='store_true',
                        help='不生成可视化图表')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    print("Loading data...")
    zh_sents, en_sents = load_data(args.num_train)
    
    # 构建词表
    zh_vocab = Vocab()
    en_vocab = Vocab()
    for zh, en in zip(zh_sents['train'], en_sents['train']):
        zh_vocab.add_sent(zh)
        en_vocab.add_sent(en)
    print(f"中文词表大小: {len(zh_vocab)}")
    print(f"英文词表大小: {len(en_vocab)}")
    
    # 创建数据加载器
    pad_id = zh_vocab.word2idx['[PAD]']
    trainloader, validloader, testloader = create_dataloader(
        zh_sents, en_sents, zh_vocab, en_vocab,
        args.max_len, args.batch_size, pad_id
    )
    
    # 创建模型
    print(f"Creating model: {args.model_type}" + 
          (" + Attention" if args.use_attention and args.model_type != 'transformer' else ""))
    model = get_model(
        model_type=args.model_type,
        use_attention=args.use_attention,
        src_vocab=zh_vocab,
        tgt_vocab=en_vocab,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        max_len=args.max_len,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout
    )
    model.to(device)
    
    # 计算模型参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {num_params:,}")
    
    # 创建优化器
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 创建损失函数（忽略PAD的损失）
    weights = torch.ones(len(en_vocab)).to(device)
    weights[en_vocab.word2idx['[PAD]']] = 0
    criterion = nn.NLLLoss(weight=weights)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置实验名称
    if args.experiment_name is None:
        attn_str = "_attn" if args.use_attention and args.model_type != 'transformer' else ""
        args.experiment_name = f"{args.model_type}{attn_str}"
    
    # 初始化日志记录器
    logger = TrainingLogger(log_dir=args.log_dir, experiment_name=args.experiment_name)
    logger.set_config(
        model_type=args.model_type,
        use_attention=args.use_attention,
        device=str(device),
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        batch_size=args.batch_size,
        max_len=args.max_len,
        lr=args.lr,
        optim=args.optim,
        num_params=num_params
    )
    
    # 恢复训练
    start_epoch = 0
    best_bleu = 0.0
    if args.resume:
        start_epoch, best_bleu = load_checkpoint(model, optimizer, args.resume, device)
    
    # 开始训练
    logger.start_training()
    print(f"\nStarting training for {args.num_epoch} epochs...")
    
    checkpoint_path = os.path.join(args.save_dir, f"{args.experiment_name}_best.pt")
    
    for epoch in range(start_epoch, args.num_epoch):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.num_epoch}")
        print(f"{'='*50}")
        
        # 训练
        train_loss = train_loop(model, optimizer, criterion, trainloader, device, args.model_type)
        
        # 验证
        hypotheses, references, valid_bleu = test_loop(
            model, validloader, en_vocab, device, args.model_type
        )
        
        # 记录日志
        current_lr = optimizer.param_groups[0]['lr']
        logger.log_epoch(epoch + 1, train_loss, valid_bleu, current_lr)
        
        # 打印示例
        print(f"Reference: {references[0]}")
        print(f"Hypothesis: {hypotheses[0]}")
        
        # 保存最优模型
        if valid_bleu > best_bleu:
            best_bleu = valid_bleu
            save_checkpoint(model, optimizer, epoch, best_bleu, checkpoint_path)
    
    # 保存训练日志
    logger.save_results()
    
    # 测试最优模型
    print(f"\n{'='*50}")
    print("Testing best model...")
    print(f"{'='*50}")
    
    # 加载最优模型
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 测试
    hypotheses, references, test_bleu = test_loop(
        model, testloader, en_vocab, device, args.model_type
    )
    print(f"\nTest BLEU: {test_bleu:.2f}")
    print(f"Reference: {references[0]}")
    print(f"Hypothesis: {hypotheses[0]}")
    
    # 生成可视化图表
    if not args.no_visualize:
        print("\nGenerating visualization...")
        visualizer = Visualizer(save_dir=args.fig_dir)
        visualizer.plot_curves(logger)
    
    print(f"\nTraining completed!")
    print(f"Best validation BLEU: {best_bleu:.2f}")
    print(f"Test BLEU: {test_bleu:.2f}")


if __name__ == '__main__':
    main()
