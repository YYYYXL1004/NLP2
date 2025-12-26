"""
Checkpoint 5: 验证RNN变体

测试LSTM、GRU、Attention的输出形状和基本功能。
"""

import torch
import torch.nn as nn
from seq2seq_models import (
    LSTMCell, GRUCell, RNNCell, Attention,
    EncoderLSTM, EncoderGRU, EncoderRNN,
    DecoderLSTM, DecoderGRU, DecoderRNN,
    DecoderLSTMWithAttention, DecoderGRUWithAttention, DecoderRNNWithAttention,
    Seq2SeqLSTM, Seq2SeqGRU, Seq2SeqRNN,
    Seq2SeqLSTMWithAttention, Seq2SeqGRUWithAttention, Seq2SeqRNNWithAttention,
    Vocab
)


def test_lstm_cell_shape():
    """测试LSTMCell输出形状 - Property 1"""
    print("=" * 50)
    print("测试 LSTMCell 输出形状")
    
    batch_size = 4
    input_size = 128
    hidden_size = 256
    
    cell = LSTMCell(input_size, hidden_size)
    
    # 创建输入
    x = torch.randn(batch_size, input_size)
    h = torch.randn(batch_size, hidden_size)
    c = torch.randn(batch_size, hidden_size)
    
    # 前向传播
    h_new, c_new = cell(x, (h, c))
    
    # 验证形状
    assert h_new.shape == (batch_size, hidden_size), \
        f"h_new shape mismatch: expected {(batch_size, hidden_size)}, got {h_new.shape}"
    assert c_new.shape == (batch_size, hidden_size), \
        f"c_new shape mismatch: expected {(batch_size, hidden_size)}, got {c_new.shape}"
    
    print(f"  输入形状: x={x.shape}, h={h.shape}, c={c.shape}")
    print(f"  输出形状: h_new={h_new.shape}, c_new={c_new.shape}")
    print("  ✓ LSTMCell 输出形状正确")
    return True


def test_gru_cell_shape():
    """测试GRUCell输出形状 - Property 1"""
    print("=" * 50)
    print("测试 GRUCell 输出形状")
    
    batch_size = 4
    input_size = 128
    hidden_size = 256
    
    cell = GRUCell(input_size, hidden_size)
    
    # 创建输入
    x = torch.randn(batch_size, input_size)
    h = torch.randn(batch_size, hidden_size)
    
    # 前向传播
    h_new = cell(x, h)
    
    # 验证形状
    assert h_new.shape == (batch_size, hidden_size), \
        f"h_new shape mismatch: expected {(batch_size, hidden_size)}, got {h_new.shape}"
    
    print(f"  输入形状: x={x.shape}, h={h.shape}")
    print(f"  输出形状: h_new={h_new.shape}")
    print("  ✓ GRUCell 输出形状正确")
    return True


def test_attention_weights_normalized():
    """测试Attention权重归一化 - Property 2"""
    print("=" * 50)
    print("测试 Attention 权重归一化")
    
    batch_size = 4
    hidden_size = 256
    src_len = 10
    
    attention = Attention(hidden_size)
    
    # 创建输入
    decoder_hidden = torch.randn(batch_size, hidden_size)
    encoder_hiddens = torch.randn(batch_size, src_len, hidden_size)
    
    # 前向传播
    context, attn_weights = attention(decoder_hidden, encoder_hiddens)
    
    # 验证权重归一化（每个batch的权重和应为1）
    weight_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones(batch_size), atol=1e-5), \
        f"Attention weights not normalized: sums = {weight_sums}"
    
    print(f"  decoder_hidden形状: {decoder_hidden.shape}")
    print(f"  encoder_hiddens形状: {encoder_hiddens.shape}")
    print(f"  attention权重形状: {attn_weights.shape}")
    print(f"  权重和: {weight_sums}")
    print("  ✓ Attention 权重正确归一化")
    return True


def test_attention_output_shape():
    """测试Attention输出形状 - Property 3"""
    print("=" * 50)
    print("测试 Attention 输出形状")
    
    batch_size = 4
    hidden_size = 256
    src_len = 10
    
    attention = Attention(hidden_size)
    
    # 创建输入
    decoder_hidden = torch.randn(batch_size, hidden_size)
    encoder_hiddens = torch.randn(batch_size, src_len, hidden_size)
    
    # 前向传播
    context, attn_weights = attention(decoder_hidden, encoder_hiddens)
    
    # 验证形状
    assert context.shape == (batch_size, hidden_size), \
        f"context shape mismatch: expected {(batch_size, hidden_size)}, got {context.shape}"
    assert attn_weights.shape == (batch_size, src_len), \
        f"attn_weights shape mismatch: expected {(batch_size, src_len)}, got {attn_weights.shape}"
    
    print(f"  context形状: {context.shape}")
    print(f"  attn_weights形状: {attn_weights.shape}")
    print("  ✓ Attention 输出形状正确")
    return True


def test_encoder_lstm():
    """测试EncoderLSTM"""
    print("=" * 50)
    print("测试 EncoderLSTM")
    
    vocab_size = 1000
    embedding_dim = 128
    hidden_size = 256
    batch_size = 4
    
    encoder = EncoderLSTM(vocab_size, embedding_dim, hidden_size)
    
    # 创建输入
    input_token = torch.randint(0, vocab_size, (batch_size,))
    h = torch.zeros(batch_size, hidden_size)
    c = torch.zeros(batch_size, hidden_size)
    
    # 前向传播
    h_new, c_new = encoder(input_token, (h, c))
    
    assert h_new.shape == (batch_size, hidden_size)
    assert c_new.shape == (batch_size, hidden_size)
    
    print(f"  输入token形状: {input_token.shape}")
    print(f"  输出h形状: {h_new.shape}, c形状: {c_new.shape}")
    print("  ✓ EncoderLSTM 正常工作")
    return True


def test_encoder_gru():
    """测试EncoderGRU"""
    print("=" * 50)
    print("测试 EncoderGRU")
    
    vocab_size = 1000
    embedding_dim = 128
    hidden_size = 256
    batch_size = 4
    
    encoder = EncoderGRU(vocab_size, embedding_dim, hidden_size)
    
    # 创建输入
    input_token = torch.randint(0, vocab_size, (batch_size,))
    h = torch.zeros(batch_size, hidden_size)
    
    # 前向传播
    h_new = encoder(input_token, h)
    
    assert h_new.shape == (batch_size, hidden_size)
    
    print(f"  输入token形状: {input_token.shape}")
    print(f"  输出h形状: {h_new.shape}")
    print("  ✓ EncoderGRU 正常工作")
    return True


def test_decoder_with_attention():
    """测试带Attention的Decoder"""
    print("=" * 50)
    print("测试 Decoder with Attention")
    
    vocab_size = 1000
    embedding_dim = 128
    hidden_size = 256
    batch_size = 4
    src_len = 10
    
    # 测试LSTM with Attention
    decoder_lstm = DecoderLSTMWithAttention(vocab_size, embedding_dim, hidden_size)
    input_token = torch.randint(0, vocab_size, (batch_size,))
    h = torch.randn(batch_size, hidden_size)
    c = torch.randn(batch_size, hidden_size)
    encoder_hiddens = torch.randn(batch_size, src_len, hidden_size)
    
    output, (h_new, c_new), attn_weights = decoder_lstm(input_token, (h, c), encoder_hiddens)
    
    assert output.shape == (batch_size, vocab_size)
    assert h_new.shape == (batch_size, hidden_size)
    assert c_new.shape == (batch_size, hidden_size)
    assert attn_weights.shape == (batch_size, src_len)
    
    print(f"  DecoderLSTMWithAttention:")
    print(f"    输出形状: {output.shape}")
    print(f"    attention权重形状: {attn_weights.shape}")
    
    # 测试GRU with Attention
    decoder_gru = DecoderGRUWithAttention(vocab_size, embedding_dim, hidden_size)
    h = torch.randn(batch_size, hidden_size)
    
    output, h_new, attn_weights = decoder_gru(input_token, h, encoder_hiddens)
    
    assert output.shape == (batch_size, vocab_size)
    assert h_new.shape == (batch_size, hidden_size)
    assert attn_weights.shape == (batch_size, src_len)
    
    print(f"  DecoderGRUWithAttention:")
    print(f"    输出形状: {output.shape}")
    print(f"    attention权重形状: {attn_weights.shape}")
    
    # 测试RNN with Attention
    decoder_rnn = DecoderRNNWithAttention(vocab_size, embedding_dim, hidden_size)
    h = torch.randn(batch_size, hidden_size)
    
    output, h_new, attn_weights = decoder_rnn(input_token, h, encoder_hiddens)
    
    assert output.shape == (batch_size, vocab_size)
    assert h_new.shape == (batch_size, hidden_size)
    assert attn_weights.shape == (batch_size, src_len)
    
    print(f"  DecoderRNNWithAttention:")
    print(f"    输出形状: {output.shape}")
    print(f"    attention权重形状: {attn_weights.shape}")
    
    print("  ✓ 所有带Attention的Decoder正常工作")
    return True


def create_dummy_vocab():
    """创建测试用的词表"""
    vocab = Vocab()
    for i in range(100):
        vocab.add_word(f"word_{i}")
    return vocab


def test_seq2seq_lstm_forward():
    """测试Seq2SeqLSTM前向传播"""
    print("=" * 50)
    print("测试 Seq2SeqLSTM 前向传播")
    
    src_vocab = create_dummy_vocab()
    tgt_vocab = create_dummy_vocab()
    embedding_dim = 64
    hidden_size = 128
    max_len = 10
    batch_size = 4
    src_len = 8
    tgt_len = 6
    
    model = Seq2SeqLSTM(src_vocab, tgt_vocab, embedding_dim, hidden_size, max_len)
    
    # 创建输入
    src = torch.randint(0, len(src_vocab), (batch_size, src_len))
    tgt = torch.randint(0, len(tgt_vocab), (batch_size, tgt_len))
    
    # 前向传播
    outputs = model(src, tgt)
    
    assert outputs.shape == (batch_size, tgt_len, len(tgt_vocab))
    
    print(f"  src形状: {src.shape}")
    print(f"  tgt形状: {tgt.shape}")
    print(f"  outputs形状: {outputs.shape}")
    print("  ✓ Seq2SeqLSTM 前向传播正常")
    return True


def test_seq2seq_gru_forward():
    """测试Seq2SeqGRU前向传播"""
    print("=" * 50)
    print("测试 Seq2SeqGRU 前向传播")
    
    src_vocab = create_dummy_vocab()
    tgt_vocab = create_dummy_vocab()
    embedding_dim = 64
    hidden_size = 128
    max_len = 10
    batch_size = 4
    src_len = 8
    tgt_len = 6
    
    model = Seq2SeqGRU(src_vocab, tgt_vocab, embedding_dim, hidden_size, max_len)
    
    # 创建输入
    src = torch.randint(0, len(src_vocab), (batch_size, src_len))
    tgt = torch.randint(0, len(tgt_vocab), (batch_size, tgt_len))
    
    # 前向传播
    outputs = model(src, tgt)
    
    assert outputs.shape == (batch_size, tgt_len, len(tgt_vocab))
    
    print(f"  src形状: {src.shape}")
    print(f"  tgt形状: {tgt.shape}")
    print(f"  outputs形状: {outputs.shape}")
    print("  ✓ Seq2SeqGRU 前向传播正常")
    return True


def test_seq2seq_with_attention_forward():
    """测试带Attention的Seq2Seq前向传播"""
    print("=" * 50)
    print("测试 Seq2Seq with Attention 前向传播")
    
    src_vocab = create_dummy_vocab()
    tgt_vocab = create_dummy_vocab()
    embedding_dim = 64
    hidden_size = 128
    max_len = 10
    batch_size = 4
    src_len = 8
    tgt_len = 6
    
    # 测试LSTM + Attention
    model_lstm = Seq2SeqLSTMWithAttention(src_vocab, tgt_vocab, embedding_dim, hidden_size, max_len)
    src = torch.randint(0, len(src_vocab), (batch_size, src_len))
    tgt = torch.randint(0, len(tgt_vocab), (batch_size, tgt_len))
    outputs = model_lstm(src, tgt)
    assert outputs.shape == (batch_size, tgt_len, len(tgt_vocab))
    print(f"  Seq2SeqLSTMWithAttention outputs形状: {outputs.shape}")
    
    # 测试GRU + Attention
    model_gru = Seq2SeqGRUWithAttention(src_vocab, tgt_vocab, embedding_dim, hidden_size, max_len)
    outputs = model_gru(src, tgt)
    assert outputs.shape == (batch_size, tgt_len, len(tgt_vocab))
    print(f"  Seq2SeqGRUWithAttention outputs形状: {outputs.shape}")
    
    # 测试RNN + Attention
    model_rnn = Seq2SeqRNNWithAttention(src_vocab, tgt_vocab, embedding_dim, hidden_size, max_len)
    outputs = model_rnn(src, tgt)
    assert outputs.shape == (batch_size, tgt_len, len(tgt_vocab))
    print(f"  Seq2SeqRNNWithAttention outputs形状: {outputs.shape}")
    
    print("  ✓ 所有带Attention的Seq2Seq前向传播正常")
    return True


def test_gradient_flow():
    """测试梯度能否正常反向传播"""
    print("=" * 50)
    print("测试 梯度反向传播")
    
    src_vocab = create_dummy_vocab()
    tgt_vocab = create_dummy_vocab()
    embedding_dim = 64
    hidden_size = 128
    max_len = 10
    batch_size = 4
    src_len = 8
    tgt_len = 6
    
    models = [
        ("Seq2SeqLSTM", Seq2SeqLSTM(src_vocab, tgt_vocab, embedding_dim, hidden_size, max_len)),
        ("Seq2SeqGRU", Seq2SeqGRU(src_vocab, tgt_vocab, embedding_dim, hidden_size, max_len)),
        ("Seq2SeqLSTMWithAttention", Seq2SeqLSTMWithAttention(src_vocab, tgt_vocab, embedding_dim, hidden_size, max_len)),
        ("Seq2SeqGRUWithAttention", Seq2SeqGRUWithAttention(src_vocab, tgt_vocab, embedding_dim, hidden_size, max_len)),
    ]
    
    criterion = nn.NLLLoss(ignore_index=tgt_vocab.index("[PAD]"))
    
    for name, model in models:
        src = torch.randint(0, len(src_vocab), (batch_size, src_len))
        tgt = torch.randint(0, len(tgt_vocab), (batch_size, tgt_len))
        
        # 前向传播
        outputs = model(src, tgt)
        
        # 计算损失
        loss = criterion(outputs.view(-1, len(tgt_vocab)), tgt.view(-1))
        
        # 反向传播
        loss.backward()
        
        # 检查梯度是否存在
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, f"{name} 没有梯度流动"
        print(f"  {name}: loss={loss.item():.4f}, 梯度正常")
        
        # 清除梯度
        model.zero_grad()
    
    print("  ✓ 所有模型梯度反向传播正常")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Checkpoint 5: 验证RNN变体")
    print("=" * 60 + "\n")
    
    tests = [
        ("LSTMCell输出形状", test_lstm_cell_shape),
        ("GRUCell输出形状", test_gru_cell_shape),
        ("Attention权重归一化", test_attention_weights_normalized),
        ("Attention输出形状", test_attention_output_shape),
        ("EncoderLSTM", test_encoder_lstm),
        ("EncoderGRU", test_encoder_gru),
        ("Decoder with Attention", test_decoder_with_attention),
        ("Seq2SeqLSTM前向传播", test_seq2seq_lstm_forward),
        ("Seq2SeqGRU前向传播", test_seq2seq_gru_forward),
        ("Seq2Seq with Attention前向传播", test_seq2seq_with_attention_forward),
        ("梯度反向传播", test_gradient_flow),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"  ✗ {name} 失败: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
