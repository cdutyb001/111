#!/usr/bin/env python3
"""
SGSNet 训练脚本

功能:
1. 支持KITTI、NYUv2等深度补全数据集
2. 混合损失函数: L1 + 梯度 + 不确定性
3. 支持断点续训
4. 内存优化: 混合精度训练

使用方法:
    # 基础训练
    python3 train_sgsnet.py --data-dir /path/to/kitti
    
    # 指定GPU和批次大小
    python3 train_sgsnet.py --data-dir /path/to/data --batch-size 4 --gpu 0
    
    # 从检查点恢复
    python3 train_sgsnet.py --data-dir /path/to/data --resume checkpoints/latest.pth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import cv2
import os
import argparse
import time
from pathlib import Path
from typing import Tuple, Dict, Optional

from sgsnet import SGSNet, SGSNetLite, SGSNetLarge


# ============ 损失函数 ============

class DepthCompletionLoss(nn.Module):
    """
    深度补全混合损失函数
    
    L = λ1 * L_depth + λ2 * L_grad + λ3 * L_uncertainty
    
    - L_depth: 深度L1损失 (仅在有GT的位置)
    - L_grad: 梯度一致性损失 (边缘对齐)
    - L_uncertainty: 不确定性正则化损失
    """
    
    def __init__(self, 
                 lambda_depth: float = 1.0,
                 lambda_grad: float = 0.5,
                 lambda_uncertainty: float = 0.1):
        super().__init__()
        self.lambda_depth = lambda_depth
        self.lambda_grad = lambda_grad
        self.lambda_uncertainty = lambda_uncertainty
        
        # Sobel算子用于梯度计算
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
    
    def compute_gradient(self, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算深度图梯度"""
        grad_x = F.conv2d(depth, self.sobel_x, padding=1)
        grad_y = F.conv2d(depth, self.sobel_y, padding=1)
        return grad_x, grad_y
    
    def forward(self, 
                pred_depth: torch.Tensor,      # (B, 1, H, W)
                pred_uncertainty: torch.Tensor, # (B, 1, H, W)
                gt_depth: torch.Tensor,         # (B, 1, H, W)
                sparse_depth: torch.Tensor      # (B, 1, H, W)
               ) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        Args:
            pred_depth: 预测的稠密深度
            pred_uncertainty: 预测的不确定性 [0,1]
            gt_depth: 真实稠密深度 (用于监督)
            sparse_depth: 稀疏深度输入
        
        Returns:
            包含各项损失的字典
        """
        # 有效深度掩码 (GT > 0)
        valid_mask = (gt_depth > 0).float()
        num_valid = valid_mask.sum() + 1e-8
        
        # ========== 1. 深度L1损失 ==========
        depth_diff = torch.abs(pred_depth - gt_depth) * valid_mask
        loss_depth = depth_diff.sum() / num_valid
        
        # ========== 2. 梯度一致性损失 ==========
        pred_grad_x, pred_grad_y = self.compute_gradient(pred_depth)
        gt_grad_x, gt_grad_y = self.compute_gradient(gt_depth)
        
        grad_diff_x = torch.abs(pred_grad_x - gt_grad_x) * valid_mask
        grad_diff_y = torch.abs(pred_grad_y - gt_grad_y) * valid_mask
        loss_grad = (grad_diff_x.sum() + grad_diff_y.sum()) / (2 * num_valid)
        
        # ========== 3. 不确定性正则化损失 ==========
        # 思路: 不确定性应该与预测误差正相关
        # 误差大的地方，不确定性应该高
        normalized_error = depth_diff / (gt_depth.clamp(min=1.0))  # 相对误差
        
        # 不确定性应该预测相对误差
        uncertainty_target = normalized_error.clamp(max=1.0).detach()
        loss_uncertainty = F.mse_loss(
            pred_uncertainty * valid_mask, 
            uncertainty_target * valid_mask,
            reduction='sum'
        ) / num_valid
        
        # ========== 总损失 ==========
        total_loss = (
            self.lambda_depth * loss_depth +
            self.lambda_grad * loss_grad +
            self.lambda_uncertainty * loss_uncertainty
        )
        
        return {
            'total': total_loss,
            'depth': loss_depth,
            'grad': loss_grad,
            'uncertainty': loss_uncertainty
        }


# ============ 数据集 ============

class DepthCompletionDataset(Dataset):
    """
    通用深度补全数据集
    
    目录结构:
        data_dir/
            rgb/          # RGB图像 (png/jpg)
            sparse/       # 稀疏深度 (png, 16bit)
            dense/        # 稠密深度GT (png, 16bit) [可选]
    """
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 height: int = 480,
                 width: int = 640,
                 max_depth: float = 100.0,
                 augment: bool = True):
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.height = height
        self.width = width
        self.max_depth = max_depth
        self.augment = augment and (split == 'train')
        
        # 查找所有样本
        rgb_dir = self.data_dir / 'rgb'
        self.samples = sorted(rgb_dir.glob('*.png')) + sorted(rgb_dir.glob('*.jpg'))
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {rgb_dir}")
        
        print(f"Found {len(self.samples)} samples in {data_dir}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rgb_path = self.samples[idx]
        stem = rgb_path.stem
        
        # 加载RGB
        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # 加载稀疏深度
        sparse_path = self.data_dir / 'sparse' / f'{stem}.png'
        if sparse_path.exists():
            sparse_depth = cv2.imread(str(sparse_path), cv2.IMREAD_UNCHANGED)
            sparse_depth = sparse_depth.astype(np.float32) / 256.0  # KITTI格式
        else:
            sparse_depth = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
        
        # 加载稠密深度GT (如果存在)
        dense_path = self.data_dir / 'dense' / f'{stem}.png'
        if dense_path.exists():
            dense_depth = cv2.imread(str(dense_path), cv2.IMREAD_UNCHANGED)
            dense_depth = dense_depth.astype(np.float32) / 256.0
        else:
            dense_depth = sparse_depth.copy()  # 自监督模式
        
        # 调整尺寸
        rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        sparse_depth = cv2.resize(sparse_depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        dense_depth = cv2.resize(dense_depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        
        # 数据增强
        if self.augment:
            # 水平翻转
            if np.random.random() > 0.5:
                rgb = np.fliplr(rgb).copy()
                sparse_depth = np.fliplr(sparse_depth).copy()
                dense_depth = np.fliplr(dense_depth).copy()
            
            # 颜色抖动
            rgb = rgb.astype(np.float32)
            rgb *= np.random.uniform(0.8, 1.2, (1, 1, 3))
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        
        # 转换为Tensor
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        sparse_depth = torch.from_numpy(sparse_depth).unsqueeze(0).float()
        dense_depth = torch.from_numpy(dense_depth).unsqueeze(0).float()
        
        # 裁剪到最大深度
        sparse_depth = sparse_depth.clamp(0, self.max_depth)
        dense_depth = dense_depth.clamp(0, self.max_depth)
        
        return {
            'rgb': rgb,
            'sparse_depth': sparse_depth,
            'dense_depth': dense_depth,
            'filename': stem
        }


class SyntheticDataset(Dataset):
    """
    合成数据集 - 用于测试训练流程
    生成随机的RGB和深度图
    """
    
    def __init__(self, 
                 num_samples: int = 1000,
                 height: int = 480,
                 width: int = 640,
                 max_depth: float = 100.0):
        self.num_samples = num_samples
        self.height = height
        self.width = width
        self.max_depth = max_depth
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 生成随机RGB
        rgb = torch.rand(3, self.height, self.width)
        
        # 生成平滑的稠密深度 (模拟真实场景)
        # 使用多个高斯核心生成平滑深度
        dense_depth = torch.zeros(1, self.height, self.width)
        for _ in range(5):
            cx, cy = np.random.randint(0, self.width), np.random.randint(0, self.height)
            sigma = np.random.uniform(50, 150)
            depth_val = np.random.uniform(5, self.max_depth)
            
            y, x = torch.meshgrid(
                torch.arange(self.height, dtype=torch.float32),
                torch.arange(self.width, dtype=torch.float32),
                indexing='ij'
            )
            gauss = torch.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
            dense_depth[0] += gauss * depth_val
        
        dense_depth = dense_depth.clamp(1, self.max_depth)
        
        # 生成稀疏深度 (随机采样约5%的点)
        sparse_mask = torch.rand(1, self.height, self.width) < 0.05
        sparse_depth = dense_depth * sparse_mask.float()
        
        return {
            'rgb': rgb,
            'sparse_depth': sparse_depth,
            'dense_depth': dense_depth,
            'filename': f'synthetic_{idx:06d}'
        }


# ============ 训练器 ============

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-4,
                 device: str = 'cuda',
                 save_dir: str = 'checkpoints',
                 use_amp: bool = True):
        
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # 学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        # 损失函数
        self.criterion = DepthCompletionLoss().to(device)
        
        # 混合精度
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # 记录
        self.epoch = 0
        self.best_loss = float('inf')
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        total_depth_loss = 0
        total_grad_loss = 0
        total_unc_loss = 0
        num_batches = 0
        
        for batch in self.train_loader:
            rgb = batch['rgb'].to(self.device)
            sparse_depth = batch['sparse_depth'].to(self.device)
            dense_depth = batch['dense_depth'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    pred_depth, pred_unc = self.model(rgb, sparse_depth)
                    losses = self.criterion(pred_depth, pred_unc, dense_depth, sparse_depth)
                
                self.scaler.scale(losses['total']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred_depth, pred_unc = self.model(rgb, sparse_depth)
                losses = self.criterion(pred_depth, pred_unc, dense_depth, sparse_depth)
                losses['total'].backward()
                self.optimizer.step()
            
            total_loss += losses['total'].item()
            total_depth_loss += losses['depth'].item()
            total_grad_loss += losses['grad'].item()
            total_unc_loss += losses['uncertainty'].item()
            num_batches += 1
        
        return {
            'total': total_loss / num_batches,
            'depth': total_depth_loss / num_batches,
            'grad': total_grad_loss / num_batches,
            'uncertainty': total_unc_loss / num_batches
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0
        total_rmse = 0
        num_batches = 0
        
        for batch in self.val_loader:
            rgb = batch['rgb'].to(self.device)
            sparse_depth = batch['sparse_depth'].to(self.device)
            dense_depth = batch['dense_depth'].to(self.device)
            
            pred_depth, pred_unc = self.model(rgb, sparse_depth)
            losses = self.criterion(pred_depth, pred_unc, dense_depth, sparse_depth)
            
            # 计算RMSE
            valid_mask = dense_depth > 0
            rmse = torch.sqrt(((pred_depth - dense_depth) ** 2 * valid_mask).sum() / valid_mask.sum())
            
            total_loss += losses['total'].item()
            total_rmse += rmse.item()
            num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_rmse': total_rmse / num_batches
        }
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss
        }
        
        torch.save(checkpoint, self.save_dir / filename)
        
        if is_best:
            torch.save(self.model.state_dict(), self.save_dir / 'best_model.pth')
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self, num_epochs: int):
        """完整训练流程"""
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"AMP: {self.use_amp}")
        print()
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # 训练
            train_losses = self.train_epoch()
            
            # 验证
            val_metrics = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            
            # 计算时间
            elapsed = time.time() - start_time
            
            # 打印日志
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Loss: {train_losses['total']:.4f} "
                  f"(D: {train_losses['depth']:.4f}, "
                  f"G: {train_losses['grad']:.4f}, "
                  f"U: {train_losses['uncertainty']:.4f}) "
                  f"LR: {lr:.6f} "
                  f"Time: {elapsed:.1f}s")
            
            if val_metrics:
                print(f"         Val Loss: {val_metrics['val_loss']:.4f}, "
                      f"RMSE: {val_metrics['val_rmse']:.3f}m")
            
            # 保存检查点
            is_best = train_losses['total'] < self.best_loss
            if is_best:
                self.best_loss = train_losses['total']
            
            self.save_checkpoint('latest.pth', is_best=is_best)
            
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch+1}.pth')
        
        print("\nTraining completed!")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"Checkpoints saved to: {self.save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train SGSNet for depth completion')
    
    # 数据
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to dataset (if not provided, uses synthetic data)')
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=640)
    
    # 模型
    parser.add_argument('--model-size', type=str, default='standard',
                        choices=['lite', 'standard', 'large'])
    parser.add_argument('--max-depth', type=float, default=100.0)
    
    # 训练
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    
    # 其他
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    
    args = parser.parse_args()
    
    # 设置GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'
    
    # 创建模型
    print("Creating model...")
    if args.model_size == 'lite':
        model = SGSNetLite(max_depth=args.max_depth)
    elif args.model_size == 'large':
        model = SGSNetLarge(max_depth=args.max_depth)
    else:
        model = SGSNet(max_depth=args.max_depth)
    
    print(model.get_memory_footprint())
    
    # 创建数据集
    print("\nCreating dataset...")
    if args.data_dir:
        train_dataset = DepthCompletionDataset(
            args.data_dir, split='train',
            height=args.height, width=args.width,
            max_depth=args.max_depth
        )
    else:
        print("No data directory provided, using synthetic data for testing")
        train_dataset = SyntheticDataset(
            num_samples=1000,
            height=args.height, width=args.width,
            max_depth=args.max_depth
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        save_dir=args.save_dir,
        use_amp=not args.no_amp
    )
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train(args.epochs)


if __name__ == "__main__":
    main()