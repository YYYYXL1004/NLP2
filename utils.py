"""
Utility classes for training logging and visualization.

包含TrainingLogger和Visualizer类，用于记录训练过程和可视化结果。
"""

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Optional
from datetime import datetime

import matplotlib.pyplot as plt


@dataclass
class EpochLog:
    """单个epoch的训练日志"""
    epoch: int
    train_loss: float
    valid_bleu: float
    learning_rate: float
    time_elapsed: float


class TrainingLogger:
    """
    训练日志记录器
    
    记录每个epoch的训练损失、验证BLEU分数、学习率和时间等信息。
    支持保存为JSON文件。
    
    Requirements: 6.1, 6.2
    """
    
    def __init__(self, log_dir: str = "logs", experiment_name: Optional[str] = None):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志保存目录
            experiment_name: 实验名称，默认使用时间戳
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.epoch_logs: List[EpochLog] = []
        self.start_time: Optional[float] = None
        self.device_info: str = ""
        self.model_type: str = ""
        self.use_attention: bool = False
        self.config: dict = {}
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
    
    def set_config(self, model_type: str, use_attention: bool, 
                   device: str, **kwargs):
        """
        设置实验配置信息
        
        Args:
            model_type: 模型类型 (rnn/lstm/gru/transformer)
            use_attention: 是否使用attention
            device: 运行设备 (cpu/cuda)
            **kwargs: 其他配置参数
        """
        self.model_type = model_type
        self.use_attention = use_attention
        self.device_info = device
        self.config = {
            "model_type": model_type,
            "use_attention": use_attention,
            "device": device,
            **kwargs
        }
    
    def start_training(self):
        """开始训练计时"""
        self.start_time = time.time()
    
    def log_epoch(self, epoch: int, train_loss: float, valid_bleu: float, 
                  learning_rate: float):
        """
        记录单个epoch的训练指标
        
        Args:
            epoch: 当前epoch编号
            train_loss: 训练损失
            valid_bleu: 验证集BLEU分数
            learning_rate: 当前学习率
        """
        time_elapsed = time.time() - self.start_time if self.start_time else 0.0
        
        log = EpochLog(
            epoch=epoch,
            train_loss=train_loss,
            valid_bleu=valid_bleu,
            learning_rate=learning_rate,
            time_elapsed=time_elapsed
        )
        self.epoch_logs.append(log)
        
        # 打印日志
        print(f"Epoch {epoch}: loss = {train_loss:.4f}, valid_bleu = {valid_bleu:.2f}, "
              f"time = {time_elapsed:.1f}s")
    
    def get_best_epoch(self) -> Optional[EpochLog]:
        """获取验证集BLEU最高的epoch"""
        if not self.epoch_logs:
            return None
        return max(self.epoch_logs, key=lambda x: x.valid_bleu)
    
    def get_losses(self) -> List[float]:
        """获取所有epoch的训练损失"""
        return [log.train_loss for log in self.epoch_logs]
    
    def get_bleu_scores(self) -> List[float]:
        """获取所有epoch的验证BLEU分数"""
        return [log.valid_bleu for log in self.epoch_logs]
    
    def get_epochs(self) -> List[int]:
        """获取所有epoch编号"""
        return [log.epoch for log in self.epoch_logs]
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        保存训练结果为JSON文件
        
        Args:
            filename: 文件名，默认使用实验名称
        
        Returns:
            保存的文件路径
        """
        if filename is None:
            filename = f"{self.experiment_name}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        
        total_time = time.time() - self.start_time if self.start_time else 0.0
        best_epoch = self.get_best_epoch()
        
        results = {
            "experiment_name": self.experiment_name,
            "config": self.config,
            "total_training_time": total_time,
            "best_epoch": asdict(best_epoch) if best_epoch else None,
            "epoch_logs": [asdict(log) for log in self.epoch_logs]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"训练结果已保存到: {filepath}")
        return filepath
    
    def load_results(self, filepath: str):
        """
        从JSON文件加载训练结果
        
        Args:
            filepath: JSON文件路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        self.experiment_name = results.get("experiment_name", "")
        self.config = results.get("config", {})
        self.model_type = self.config.get("model_type", "")
        self.use_attention = self.config.get("use_attention", False)
        self.device_info = self.config.get("device", "")
        
        self.epoch_logs = [
            EpochLog(**log) for log in results.get("epoch_logs", [])
        ]


class Visualizer:
    """
    训练可视化工具
    
    绘制训练损失曲线和BLEU分数曲线，支持保存图表到文件。
    
    Requirements: 6.3, 6.4, 6.5
    """
    
    def __init__(self, save_dir: str = "figures"):
        """
        初始化可视化工具
        
        Args:
            save_dir: 图表保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_loss_curve(self, logger: TrainingLogger, 
                        save_path: Optional[str] = None,
                        show: bool = False) -> str:
        """
        绘制训练损失曲线
        
        Args:
            logger: TrainingLogger实例
            save_path: 保存路径，默认自动生成
            show: 是否显示图表
        
        Returns:
            保存的文件路径
        """
        epochs = logger.get_epochs()
        losses = logger.get_losses()
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses, 'b-o', linewidth=2, markersize=6)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Training Loss', fontsize=12)
        plt.title(f'Training Loss Curve - {logger.model_type}' + 
                  (' + Attention' if logger.use_attention else ''), fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(
                self.save_dir, 
                f"loss_{logger.experiment_name}.png"
            )
        
        plt.savefig(save_path, dpi=150)
        print(f"损失曲线已保存到: {save_path}")
        
        if show:
            plt.show()
        plt.close()
        
        return save_path
    
    def plot_bleu_curve(self, logger: TrainingLogger,
                        save_path: Optional[str] = None,
                        show: bool = False) -> str:
        """
        绘制BLEU分数曲线
        
        Args:
            logger: TrainingLogger实例
            save_path: 保存路径，默认自动生成
            show: 是否显示图表
        
        Returns:
            保存的文件路径
        """
        epochs = logger.get_epochs()
        bleu_scores = logger.get_bleu_scores()
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, bleu_scores, 'g-o', linewidth=2, markersize=6)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('BLEU Score', fontsize=12)
        plt.title(f'Validation BLEU Curve - {logger.model_type}' + 
                  (' + Attention' if logger.use_attention else ''), fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(
                self.save_dir, 
                f"bleu_{logger.experiment_name}.png"
            )
        
        plt.savefig(save_path, dpi=150)
        print(f"BLEU曲线已保存到: {save_path}")
        
        if show:
            plt.show()
        plt.close()
        
        return save_path
    
    def plot_curves(self, logger: TrainingLogger,
                    save_path: Optional[str] = None,
                    show: bool = False) -> str:
        """
        绘制训练损失和BLEU分数的组合图
        
        Args:
            logger: TrainingLogger实例
            save_path: 保存路径，默认自动生成
            show: 是否显示图表
        
        Returns:
            保存的文件路径
        """
        epochs = logger.get_epochs()
        losses = logger.get_losses()
        bleu_scores = logger.get_bleu_scores()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 损失曲线
        ax1.plot(epochs, losses, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Training Loss', fontsize=12)
        ax1.set_title('Training Loss', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # BLEU曲线
        ax2.plot(epochs, bleu_scores, 'g-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('BLEU Score', fontsize=12)
        ax2.set_title('Validation BLEU', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # 总标题
        model_name = logger.model_type.upper()
        if logger.use_attention:
            model_name += ' + Attention'
        fig.suptitle(f'Training Curves - {model_name}', fontsize=16, y=1.02)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(
                self.save_dir, 
                f"curves_{logger.experiment_name}.png"
            )
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
        
        if show:
            plt.show()
        plt.close()
        
        return save_path
    
    def plot_comparison(self, loggers: List[TrainingLogger],
                        labels: Optional[List[str]] = None,
                        save_path: Optional[str] = None,
                        show: bool = False) -> str:
        """
        绘制多个模型的对比图
        
        Args:
            loggers: TrainingLogger实例列表
            labels: 模型标签列表
            save_path: 保存路径
            show: 是否显示图表
        
        Returns:
            保存的文件路径
        """
        if labels is None:
            labels = [
                f"{l.model_type}" + (" + Attn" if l.use_attention else "")
                for l in loggers
            ]
        
        colors = ['b', 'r', 'g', 'orange', 'purple', 'brown']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        for i, (logger, label) in enumerate(zip(loggers, labels)):
            color = colors[i % len(colors)]
            epochs = logger.get_epochs()
            
            # 损失曲线
            ax1.plot(epochs, logger.get_losses(), f'{color}-o', 
                     linewidth=2, markersize=4, label=label)
            
            # BLEU曲线
            ax2.plot(epochs, logger.get_bleu_scores(), f'{color}-o', 
                     linewidth=2, markersize=4, label=label)
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Training Loss', fontsize=12)
        ax1.set_title('Training Loss Comparison', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('BLEU Score', fontsize=12)
        ax2.set_title('Validation BLEU Comparison', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle('Model Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, "model_comparison.png")
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"对比图已保存到: {save_path}")
        
        if show:
            plt.show()
        plt.close()
        
        return save_path
