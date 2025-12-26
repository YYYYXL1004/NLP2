"""
训练脚本 - 支持RNN/LSTM/GRU/Transformer
"""

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sacrebleu.metrics import BLEU

from seq2seq_models import (
    Vocab, load_data, create_dataloader,
    Seq2SeqRNN, Seq2SeqLSTM, Seq2SeqGRU,
    Seq2SeqRNNWithAttention, Seq2SeqLSTMWithAttention, Seq2SeqGRUWithAttention
)
from transformer import TransformerSeq2Seq
from utils import TrainingLogger, Visualizer


def get_model(args, src_vocab, tgt_vocab):
    """创建模型"""
    if args.model_type == 'transformer':
        return TransformerSeq2Seq(
            len(src_vocab), len(tgt_vocab),
            d_model=args.embedding_dim, num_heads=args.num_heads,
            num_layers=args.num_layers, d_ff=args.d_ff,
            dropout=args.dropout, max_len=args.max_len + 2,
            pad_idx=src_vocab.word2idx['[PAD]'],
            bos_idx=src_vocab.word2idx['[BOS]'],
            eos_idx=src_vocab.word2idx['[EOS]']
        )
    
    models = {
        'rnn': (Seq2SeqRNN, Seq2SeqRNNWithAttention),
        'lstm': (Seq2SeqLSTM, Seq2SeqLSTMWithAttention),
        'gru': (Seq2SeqGRU, Seq2SeqGRUWithAttention),
    }
    cls = models[args.model_type][1 if args.use_attention else 0]
    return cls(src_vocab, tgt_vocab, args.embedding_dim, args.hidden_size, args.max_len + 2)


def train_epoch(model, optimizer, criterion, loader, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    for src, tgt in tqdm(loader, desc="Training"):
        src, tgt = src.to(device), tgt.to(device)
        out = model(src, tgt)
        loss = criterion(out[:, :-1, :].reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, tgt_vocab, device):
    """评估模型"""
    model.eval()
    bleu = BLEU(force=True)
    hyps, refs = [], []
    
    for src, tgt in tqdm(loader, desc="Evaluating"):
        for i in range(len(src)):
            with torch.no_grad():
                pred = model.predict(src[i:i+1].to(device))
            ref = " ".join(tgt_vocab.decode(tgt[i].tolist(), strip=True))
            hyp = " ".join(tgt_vocab.decode(pred[0].cpu().tolist(), strip=True))
            refs.append(ref)
            hyps.append(hyp)
    
    return hyps, refs, bleu.corpus_score(hyps, [refs]).score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='lstm', choices=['rnn', 'lstm', 'gru', 'transformer'])
    parser.add_argument('--use_attention', action='store_true')
    parser.add_argument('--num_train', type=int, default=-1)
    parser.add_argument('--max_len', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 加载数据
    zh_sents, en_sents = load_data(args.num_train)
    
    # 构建词表
    zh_vocab, en_vocab = Vocab(), Vocab()
    for zh, en in zip(zh_sents['train'], en_sents['train']):
        zh_vocab.add_sent(zh)
        en_vocab.add_sent(en)
    print(f"词表大小: 中文{len(zh_vocab)}, 英文{len(en_vocab)}")
    
    # 数据加载器
    pad_id = zh_vocab.word2idx['[PAD]']
    trainloader, validloader, testloader = create_dataloader(
        zh_sents, en_sents, zh_vocab, en_vocab, args.max_len, args.batch_size, pad_id
    )
    
    # 创建模型
    model_name = args.model_type + ("_attn" if args.use_attention and args.model_type != 'transformer' else "")
    print(f"模型: {model_name}")
    model = get_model(args, zh_vocab, en_vocab).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 优化器和损失
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    weights = torch.ones(len(en_vocab), device=device)
    weights[en_vocab.word2idx['[PAD]']] = 0
    criterion = nn.NLLLoss(weight=weights)
    
    # 日志
    os.makedirs(args.save_dir, exist_ok=True)
    logger = TrainingLogger(experiment_name=model_name) if 'TrainingLogger' in dir() else None
    
    # 训练
    best_bleu = 0
    ckpt_path = os.path.join(args.save_dir, f"{model_name}_best.pt")
    
    for epoch in range(args.num_epoch):
        print(f"\n=== Epoch {epoch+1}/{args.num_epoch} ===")
        train_loss = train_epoch(model, optimizer, criterion, trainloader, device)
        hyps, refs, valid_bleu = evaluate(model, validloader, en_vocab, device)
        
        print(f"Loss: {train_loss:.4f}, BLEU: {valid_bleu:.2f}")
        print(f"Ref: {refs[0]}")
        print(f"Hyp: {hyps[0]}")
        
        if valid_bleu > best_bleu:
            best_bleu = valid_bleu
            torch.save(model.state_dict(), ckpt_path)
            print(f"保存模型到 {ckpt_path}")
    
    # 测试
    print("\n=== 测试 ===")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    hyps, refs, test_bleu = evaluate(model, testloader, en_vocab, device)
    print(f"Test BLEU: {test_bleu:.2f}")
    print(f"Best Valid BLEU: {best_bleu:.2f}")


if __name__ == '__main__':
    main()
