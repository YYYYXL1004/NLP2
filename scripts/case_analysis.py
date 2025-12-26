"""
案例分析 - 对比不同模型的翻译输出
"""
import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seq2seq_models import (
    Vocab, load_data,
    Seq2SeqRNN, Seq2SeqLSTM, Seq2SeqGRU,
    Seq2SeqRNNWithAttention, Seq2SeqLSTMWithAttention, Seq2SeqGRUWithAttention
)
from transformer import TransformerSeq2Seq


def load_model(model_type, use_attention, zh_vocab, en_vocab, device):
    """加载训练好的模型"""
    emb_dim, hidden_size, max_len = 256, 256, 12
    
    if model_type == 'transformer':
        model = TransformerSeq2Seq(
            len(zh_vocab), len(en_vocab),
            d_model=256, num_heads=8, num_layers=3, d_ff=512,
            dropout=0.1, max_len=max_len,
            pad_idx=en_vocab.word2idx['[PAD]'],
            bos_idx=en_vocab.word2idx['[BOS]'],
            eos_idx=en_vocab.word2idx['[EOS]']
        )
        ckpt_name = 'transformer'
    else:
        models = {
            'rnn': (Seq2SeqRNN, Seq2SeqRNNWithAttention),
            'lstm': (Seq2SeqLSTM, Seq2SeqLSTMWithAttention),
            'gru': (Seq2SeqGRU, Seq2SeqGRUWithAttention),
        }
        cls = models[model_type][1 if use_attention else 0]
        model = cls(zh_vocab, en_vocab, emb_dim, hidden_size, max_len)
        ckpt_name = model_type + ('_attn' if use_attention else '')
    
    ckpt_path = f'checkpoints/{ckpt_name}_best.pt'
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"加载模型: {ckpt_path}")
    else:
        print(f"警告: 未找到 {ckpt_path}")
    
    return model.to(device).eval()


def translate(model, src_sent, zh_vocab, en_vocab, device, max_len=10):
    """翻译单个句子"""
    # 编码
    src_ids = zh_vocab.encode(src_sent.split(), max_len)
    src_tensor = torch.LongTensor([src_ids]).to(device)
    
    with torch.no_grad():
        pred = model.predict(src_tensor)
    
    # 解码
    pred_words = en_vocab.decode(pred[0].cpu().tolist(), strip=True)
    return ' '.join(pred_words)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # 加载数据和词表
    zh_sents, en_sents = load_data(-1)
    zh_vocab, en_vocab = Vocab(), Vocab()
    for zh, en in zip(zh_sents['train'], en_sents['train']):
        zh_vocab.add_sent(zh)
        en_vocab.add_sent(en)
    
    # 测试样例 (从测试集选几个)
    test_cases = [
        ("我 爱 你", "I love you"),
        ("他 是 我 的 朋友", "He is my friend"),
        ("今天 天气 很 好", "The weather is nice today"),
        ("你 在 做 什么", "What are you doing"),
        ("我 想 回家", "I want to go home"),
    ]
    
    # 也从测试集随机选几个
    import random
    random.seed(42)
    for i in random.sample(range(len(zh_sents['test'])), 5):
        zh = ' '.join(zh_sents['test'][i])
        en = ' '.join(en_sents['test'][i])
        test_cases.append((zh, en))
    
    # 加载所有模型
    models = {}
    for name, (mtype, att) in [
        ('RNN', ('rnn', False)),
        ('LSTM', ('lstm', False)),
        ('GRU', ('gru', False)),
        ('RNN+Att', ('rnn', True)),
        ('LSTM+Att', ('lstm', True)),
        ('GRU+Att', ('gru', True)),
        ('Transformer', ('transformer', False)),
    ]:
        try:
            models[name] = load_model(mtype, att, zh_vocab, en_vocab, device)
        except Exception as e:
            print(f"加载 {name} 失败: {e}")
    
    print("\n" + "="*80)
    print("案例分析 - 翻译对比")
    print("="*80)
    
    results = []
    for zh, ref in test_cases:
        print(f"\n源句: {zh}")
        print(f"参考: {ref}")
        print("-" * 40)
        
        case_result = {'src': zh, 'ref': ref, 'hyps': {}}
        for name, model in models.items():
            try:
                hyp = translate(model, zh, zh_vocab, en_vocab, device)
                print(f"{name:12s}: {hyp}")
                case_result['hyps'][name] = hyp
            except Exception as e:
                print(f"{name:12s}: [错误] {e}")
        results.append(case_result)
    
    # 保存结果
    import json
    with open('figures/case_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果保存到: figures/case_analysis.json")


if __name__ == '__main__':
    main()
