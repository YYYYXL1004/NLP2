"""
训练日志和可视化工具
"""

import json
import os
import sys
import time
import matplotlib.pyplot as plt


class TrainingLogger:
    """训练日志记录 (同时输出到终端和文件)"""
    def __init__(self, log_dir="logs", name=None):
        self.log_dir = log_dir
        self.name = name or time.strftime("%Y%m%d_%H%M%S")
        self.logs = []
        self.start_time = None
        self.model_type = ""
        self.use_attention = False
        self.config = {}
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志文件，同时输出到终端和文件
        self.log_file = os.path.join(log_dir, f"{self.name}.log")
        self.file_handle = open(self.log_file, 'w', encoding='utf-8')
        self._original_stdout = sys.stdout
        sys.stdout = self  # 重定向stdout
    
    def write(self, text):
        """同时写入终端和文件"""
        self._original_stdout.write(text)
        self.file_handle.write(text)
        self.file_handle.flush()
    
    def flush(self):
        self._original_stdout.flush()
        self.file_handle.flush()
    
    def set_config(self, model_type, use_attention, device, **kwargs):
        self.model_type = model_type
        self.use_attention = use_attention
        self.config = {"model_type": model_type, "use_attention": use_attention, "device": device, **kwargs}
        print(f"配置: {self.config}")
    
    def start_training(self):
        self.start_time = time.time()
        print(f"开始训练: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def log_epoch(self, epoch, train_loss, valid_bleu, lr):
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.logs.append({"epoch": epoch, "loss": train_loss, "bleu": valid_bleu, "lr": lr, "time": elapsed})
        print(f"Loss: {train_loss:.4f}, BLEU: {valid_bleu:.2f}, 累计时间: {elapsed:.1f}s")
    
    def get_losses(self):
        return [x["loss"] for x in self.logs]
    
    def get_bleu_scores(self):
        return [x["bleu"] for x in self.logs]
    
    def get_epochs(self):
        return [x["epoch"] for x in self.logs]
    
    def save_results(self, filename=None):
        if filename is None:
            filename = f"{self.name}.json"
        path = os.path.join(self.log_dir, filename)
        data = {"name": self.name, "config": self.config, "logs": self.logs}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"结果保存到: {path}")
        print(f"日志保存到: {self.log_file}")
        return path
    
    def close(self):
        """恢复stdout并关闭文件"""
        sys.stdout = self._original_stdout
        self.file_handle.close()
    
    def __del__(self):
        try:
            self.close()
        except:
            pass


class Visualizer:
    """可视化工具"""
    def __init__(self, save_dir="figures"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_curves(self, logger, save_path=None, show=False):
        epochs = logger.get_epochs()
        losses = logger.get_losses()
        bleus = logger.get_bleu_scores()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(epochs, losses, 'b-o')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(epochs, bleus, 'g-o')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('BLEU')
        ax2.set_title('Validation BLEU')
        ax2.grid(True, alpha=0.3)
        
        title = logger.model_type.upper()
        if logger.use_attention:
            title += ' + Attention'
        fig.suptitle(title)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, f"curves_{logger.name}.png")
        plt.savefig(save_path, dpi=150)
        print(f"图表保存到: {save_path}")
        
        if show:
            plt.show()
        plt.close()
        return save_path
