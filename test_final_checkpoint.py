"""
Final Checkpoint (Task 9): 完整测试

验证所有模型配置能正常运行：
- 运行所有模型配置 (RNN, LSTM, GRU, Transformer, 带/不带Attention)
- 确保CPU和GPU都能正常运行
- 验证日志和可视化输出
"""

import os
import sys
import torch
import torch.nn as nn
import tempfile
import shutil

# 导入模型
from seq2seq_models import (
    Vocab, LSTMCell, GRUCell, RNNCell, Attention,
    Seq2SeqRNN, Seq2SeqLSTM, Seq2SeqGRU,
    Seq2SeqRNNWithAttention, Seq2SeqLSTMWithAttention, Seq2SeqGRUWithAttention
)
from transformer import TransformerSeq2Seq
from utils import TrainingLogger, Visualizer
from train import get_model


def create_dummy_vocab(size=100):
    """创建测试用的词表"""
    vocab = Vocab()
    for i in range(size):
        vocab.add_word(f"word_{i}")
    return vocab


def test_model_on_device(model, device, src_vocab, tgt_vocab, model_name):
    """在指定设备上测试模型"""
    batch_size = 4
    src_len = 8
    tgt_len = 6
    
    model = model.to(device)
    model.train()
    
    # 创建输入
    src = torch.randint(4, len(src_vocab), (batch_size, src_len)).to(device)
    tgt = torch.randint(4, len(tgt_vocab), (batch_size, tgt_len)).to(device)
    
    # 前向传播
    outputs = model(src, tgt)
    
    # 验证输出形状
    expected_shape = (batch_size, tgt_len, len(tgt_vocab))
    assert outputs.shape == expected_shape, \
        f"{model_name} output shape mismatch on {device}: expected {expected_shape}, got {outputs.shape}"
    
    # 测试反向传播
    criterion = nn.NLLLoss(ignore_index=tgt_vocab.index("[PAD]"))
    loss = criterion(outputs.view(-1, len(tgt_vocab)), tgt.view(-1))
    loss.backward()
    
    # 测试预测
    model.eval()
    with torch.no_grad():
        preds = model.predict(src[:1])
    
    assert preds.shape[0] == 1, f"{model_name} predict batch size mismatch"
    
    return True


def test_all_model_configurations():
    """测试所有模型配置"""
    print("=" * 60)
    print("测试所有模型配置")
    print("=" * 60)
    
    src_vocab = create_dummy_vocab()
    tgt_vocab = create_dummy_vocab()
    embedding_dim = 64
    hidden_size = 128
    max_len = 12
    
    # 定义所有模型配置
    model_configs = [
        ("RNN", Seq2SeqRNN(src_vocab, tgt_vocab, embedding_dim, hidden_size, max_len)),
        ("LSTM", Seq2SeqLSTM(src_vocab, tgt_vocab, embedding_dim, hidden_size, max_len)),
        ("GRU", Seq2SeqGRU(src_vocab, tgt_vocab, embedding_dim, hidden_size, max_len)),
        ("RNN+Attention", Seq2SeqRNNWithAttention(src_vocab, tgt_vocab, embedding_dim, hidden_size, max_len)),
        ("LSTM+Attention", Seq2SeqLSTMWithAttention(src_vocab, tgt_vocab, embedding_dim, hidden_size, max_len)),
        ("GRU+Attention", Seq2SeqGRUWithAttention(src_vocab, tgt_vocab, embedding_dim, hidden_size, max_len)),
        ("Transformer", TransformerSeq2Seq(
            len(src_vocab), len(tgt_vocab), d_model=64, num_heads=4, 
            num_layers=2, d_ff=128, dropout=0.1, max_len=max_len,
            pad_idx=src_vocab.word2idx['[PAD]'],
            bos_idx=src_vocab.word2idx['[BOS]'],
            eos_idx=src_vocab.word2idx['[EOS]']
        )),
    ]
    
    passed = 0
    failed = 0
    
    for name, model in model_configs:
        try:
            test_model_on_device(model, torch.device('cpu'), src_vocab, tgt_vocab, name)
            print(f"  ✓ {name} - CPU 测试通过")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name} - CPU 测试失败: {e}")
            failed += 1
    
    print(f"\n模型配置测试: {passed} 通过, {failed} 失败")
    return failed == 0


def test_gpu_if_available():
    """测试GPU运行（如果可用）"""
    print("\n" + "=" * 60)
    print("测试GPU运行")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("  CUDA不可用，跳过GPU测试")
        return True
    
    device = torch.device('cuda')
    print(f"  检测到GPU: {torch.cuda.get_device_name(0)}")
    
    src_vocab = create_dummy_vocab()
    tgt_vocab = create_dummy_vocab()
    embedding_dim = 64
    hidden_size = 128
    max_len = 12
    
    # 测试几个关键模型
    model_configs = [
        ("LSTM+Attention", Seq2SeqLSTMWithAttention(src_vocab, tgt_vocab, embedding_dim, hidden_size, max_len)),
        ("Transformer", TransformerSeq2Seq(
            len(src_vocab), len(tgt_vocab), d_model=64, num_heads=4,
            num_layers=2, d_ff=128, dropout=0.1, max_len=max_len,
            pad_idx=src_vocab.word2idx['[PAD]'],
            bos_idx=src_vocab.word2idx['[BOS]'],
            eos_idx=src_vocab.word2idx['[EOS]']
        )),
    ]
    
    passed = 0
    failed = 0
    
    for name, model in model_configs:
        try:
            test_model_on_device(model, device, src_vocab, tgt_vocab, name)
            print(f"  ✓ {name} - GPU 测试通过")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name} - GPU 测试失败: {e}")
            failed += 1
    
    print(f"\nGPU测试: {passed} 通过, {failed} 失败")
    return failed == 0


def test_get_model_function():
    """测试train.py中的get_model函数"""
    print("\n" + "=" * 60)
    print("测试 get_model 函数")
    print("=" * 60)
    
    src_vocab = create_dummy_vocab()
    tgt_vocab = create_dummy_vocab()
    
    configs = [
        ('rnn', False),
        ('rnn', True),
        ('lstm', False),
        ('lstm', True),
        ('gru', False),
        ('gru', True),
        ('transformer', False),
    ]
    
    passed = 0
    failed = 0
    
    for model_type, use_attention in configs:
        try:
            model = get_model(
                model_type=model_type,
                use_attention=use_attention,
                src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                embedding_dim=64,
                hidden_size=128,
                max_len=10
            )
            
            # 验证模型类型
            if model_type == 'transformer':
                assert isinstance(model, TransformerSeq2Seq)
            elif model_type == 'lstm':
                if use_attention:
                    assert isinstance(model, Seq2SeqLSTMWithAttention)
                else:
                    assert isinstance(model, Seq2SeqLSTM)
            elif model_type == 'gru':
                if use_attention:
                    assert isinstance(model, Seq2SeqGRUWithAttention)
                else:
                    assert isinstance(model, Seq2SeqGRU)
            else:  # rnn
                if use_attention:
                    assert isinstance(model, Seq2SeqRNNWithAttention)
                else:
                    assert isinstance(model, Seq2SeqRNN)
            
            attn_str = "+Attention" if use_attention and model_type != 'transformer' else ""
            print(f"  ✓ get_model({model_type}, attention={use_attention}) -> {type(model).__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ get_model({model_type}, attention={use_attention}) 失败: {e}")
            failed += 1
    
    print(f"\nget_model测试: {passed} 通过, {failed} 失败")
    return failed == 0


def test_logger_functionality():
    """测试日志记录功能"""
    print("\n" + "=" * 60)
    print("测试日志记录功能")
    print("=" * 60)
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        logger = TrainingLogger(log_dir=temp_dir, experiment_name="test_experiment")
        
        # 设置配置
        logger.set_config(
            model_type="lstm",
            use_attention=True,
            device="cpu",
            embedding_dim=256,
            hidden_size=256
        )
        
        # 开始训练
        logger.start_training()
        
        # 记录几个epoch
        for epoch in range(1, 4):
            logger.log_epoch(
                epoch=epoch,
                train_loss=5.0 - epoch * 0.5,
                valid_bleu=epoch * 5.0,
                learning_rate=0.001
            )
        
        # 验证日志记录
        assert len(logger.epoch_logs) == 3, "Epoch logs count mismatch"
        assert logger.get_losses() == [4.5, 4.0, 3.5], "Losses mismatch"
        assert logger.get_bleu_scores() == [5.0, 10.0, 15.0], "BLEU scores mismatch"
        
        # 获取最佳epoch
        best = logger.get_best_epoch()
        assert best.epoch == 3, "Best epoch should be 3"
        assert best.valid_bleu == 15.0, "Best BLEU should be 15.0"
        
        # 保存结果
        filepath = logger.save_results()
        assert os.path.exists(filepath), "Log file not created"
        
        # 加载并验证
        logger2 = TrainingLogger()
        logger2.load_results(filepath)
        assert len(logger2.epoch_logs) == 3, "Loaded logs count mismatch"
        
        print("  ✓ 日志记录功能正常")
        print("  ✓ 日志保存和加载功能正常")
        return True
        
    except Exception as e:
        print(f"  ✗ 日志功能测试失败: {e}")
        return False
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)



def test_visualizer_functionality():
    """测试可视化功能"""
    print("\n" + "=" * 60)
    print("测试可视化功能")
    print("=" * 60)
    
    # 创建临时目录
    temp_log_dir = tempfile.mkdtemp()
    temp_fig_dir = tempfile.mkdtemp()
    
    try:
        # 创建logger并记录数据
        logger = TrainingLogger(log_dir=temp_log_dir, experiment_name="viz_test")
        logger.set_config(model_type="lstm", use_attention=True, device="cpu")
        logger.start_training()
        
        for epoch in range(1, 6):
            logger.log_epoch(
                epoch=epoch,
                train_loss=5.0 - epoch * 0.3,
                valid_bleu=epoch * 4.0,
                learning_rate=0.001
            )
        
        # 创建可视化器
        visualizer = Visualizer(save_dir=temp_fig_dir)
        
        # 测试绘制损失曲线
        loss_path = visualizer.plot_loss_curve(logger)
        assert os.path.exists(loss_path), "Loss curve not saved"
        print(f"  ✓ 损失曲线已保存: {os.path.basename(loss_path)}")
        
        # 测试绘制BLEU曲线
        bleu_path = visualizer.plot_bleu_curve(logger)
        assert os.path.exists(bleu_path), "BLEU curve not saved"
        print(f"  ✓ BLEU曲线已保存: {os.path.basename(bleu_path)}")
        
        # 测试绘制组合曲线
        curves_path = visualizer.plot_curves(logger)
        assert os.path.exists(curves_path), "Combined curves not saved"
        print(f"  ✓ 组合曲线已保存: {os.path.basename(curves_path)}")
        
        # 测试多模型对比图
        logger2 = TrainingLogger(log_dir=temp_log_dir, experiment_name="viz_test2")
        logger2.set_config(model_type="gru", use_attention=False, device="cpu")
        logger2.start_training()
        for epoch in range(1, 6):
            logger2.log_epoch(epoch, 4.5 - epoch * 0.25, epoch * 3.5, 0.001)
        
        comparison_path = visualizer.plot_comparison([logger, logger2])
        assert os.path.exists(comparison_path), "Comparison plot not saved"
        print(f"  ✓ 对比图已保存: {os.path.basename(comparison_path)}")
        
        print("  ✓ 可视化功能正常")
        return True
        
    except Exception as e:
        print(f"  ✗ 可视化功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理临时目录
        shutil.rmtree(temp_log_dir)
        shutil.rmtree(temp_fig_dir)


def test_checkpoint_save_load():
    """测试模型checkpoint保存和加载"""
    print("\n" + "=" * 60)
    print("测试模型checkpoint保存和加载")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        src_vocab = create_dummy_vocab()
        tgt_vocab = create_dummy_vocab()
        
        # 创建模型
        model = Seq2SeqLSTMWithAttention(src_vocab, tgt_vocab, 64, 128, 12)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 保存checkpoint
        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")
        checkpoint = {
            'epoch': 5,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_bleu': 25.5
        }
        torch.save(checkpoint, checkpoint_path)
        
        assert os.path.exists(checkpoint_path), "Checkpoint not saved"
        print(f"  ✓ Checkpoint已保存")
        
        # 创建新模型并加载
        model2 = Seq2SeqLSTMWithAttention(src_vocab, tgt_vocab, 64, 128, 12)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        
        loaded = torch.load(checkpoint_path)
        model2.load_state_dict(loaded['model_state_dict'])
        optimizer2.load_state_dict(loaded['optimizer_state_dict'])
        
        assert loaded['epoch'] == 5, "Epoch mismatch"
        assert loaded['best_bleu'] == 25.5, "Best BLEU mismatch"
        print(f"  ✓ Checkpoint已加载: epoch={loaded['epoch']}, best_bleu={loaded['best_bleu']}")
        
        # 验证模型参数一致
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.equal(p1, p2), f"Parameter {n1} mismatch"
        
        print("  ✓ 模型参数一致性验证通过")
        return True
        
    except Exception as e:
        print(f"  ✗ Checkpoint测试失败: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir)


def test_training_loop_simulation():
    """模拟完整训练循环"""
    print("\n" + "=" * 60)
    print("模拟完整训练循环")
    print("=" * 60)
    
    src_vocab = create_dummy_vocab()
    tgt_vocab = create_dummy_vocab()
    
    # 创建模型
    model = Seq2SeqLSTMWithAttention(src_vocab, tgt_vocab, 64, 128, 12)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    weights = torch.ones(len(tgt_vocab))
    weights[tgt_vocab.word2idx['[PAD]']] = 0
    criterion = nn.NLLLoss(weight=weights)
    
    # 创建模拟数据
    batch_size = 8
    src_len = 10
    tgt_len = 8
    
    src = torch.randint(4, len(src_vocab), (batch_size, src_len))
    tgt = torch.randint(4, len(tgt_vocab), (batch_size, tgt_len))
    
    # 模拟训练
    model.train()
    losses = []
    
    for epoch in range(3):
        optimizer.zero_grad()
        outputs = model(src, tgt)
        loss = criterion(
            outputs[:, :-1, :].reshape(-1, outputs.shape[-1]),
            tgt[:, 1:].reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        print(f"  Epoch {epoch + 1}: loss = {loss.item():.4f}")
    
    # 模拟评估
    model.eval()
    with torch.no_grad():
        preds = model.predict(src[:1])
    
    print(f"  预测序列长度: {preds.shape[1]}")
    print("  ✓ 训练循环模拟完成")
    return True


def run_all_tests():
    """运行所有Final Checkpoint测试"""
    print("\n" + "=" * 70)
    print("Final Checkpoint (Task 9): 完整测试")
    print("=" * 70 + "\n")
    
    tests = [
        ("所有模型配置测试", test_all_model_configurations),
        ("GPU测试", test_gpu_if_available),
        ("get_model函数测试", test_get_model_function),
        ("日志记录功能测试", test_logger_functionality),
        ("可视化功能测试", test_visualizer_functionality),
        ("Checkpoint保存加载测试", test_checkpoint_save_load),
        ("训练循环模拟", test_training_loop_simulation),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ {name} 异常: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Final Checkpoint 测试结果: {passed} 通过, {failed} 失败")
    print("=" * 70)
    
    if failed == 0:
        print("\n✓ 所有测试通过！Final Checkpoint 验证成功。")
    else:
        print(f"\n✗ 有 {failed} 个测试失败，请检查。")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
