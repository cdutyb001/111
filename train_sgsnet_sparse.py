#!/usr/bin/env python3
"""
SGSNet训练脚本 - 稀疏深度监督版本
专门用于R3live等只有稀疏LiDAR深度的数据集

与标准版本的区别:
1. 只使用稀疏深度作为监督
2. 损失函数只在有稀疏点的位置计算
3. 添加平滑约束和梯度约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import cv2
import argparse
import time
from pathlib import Path
from typing import Dict

from sgsnet import SGSNet, SGSNetLite, SGSNetLarge


class SparseDepthLoss(nn.Module):
    """
    稀疏深度监督损失
    
    L = λ1 * L_sparse + λ2 * L_smooth + λ3 * L_uncertainty
    
    - L_sparse: 稀疏点处的L1损失
    - L_smooth: 深度平滑约束（边缘感知）
    - L_uncertainty: 不确定性正则化
    """
    
    def __init__(self, 
                 lambda_sparse: float = 1.0,
                 lambda_smooth: float = 0.1,
                 lambda_uncertainty: float = 0.05):
        super().__init__()
        self.lambda_sparse = lambda_sparse
        self.lambda_smooth = lambda_smooth
        self.lambda_uncertainty = lambda_uncertainty
        
        # Sobel算子
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1], [0, 0, 0], [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
    
    def forward(self, pred_depth, pred_unc, sparse_depth, rgb):
        """
        Args:
            pred_depth: (B, 1, H, W) 预测稠密深度（注意：模型已经在稀疏点处强制=sparse）
            pred_unc: (B, 1, H, W) 预测不确定性
            sparse_depth: (B, 1, H, W) 稀疏深度GT
            rgb: (B, 3, H, W) RGB图像（用于边缘检测）
        """
        # 稀疏点掩码
        valid_mask = (sparse_depth > 0).float()
        num_valid = valid_mask.sum() + 1e-8
        
        # ========== 1. 稀疏深度损失 ==========
        # 只在有LiDAR观测的位置计算
        sparse_diff = torch.abs(pred_depth - sparse_depth) * valid_mask
        loss_sparse = sparse_diff.sum() / num_valid
        
        # ========== 2. 平滑约束损失 ==========
        # 深度图应该平滑，但在RGB边缘处允许不连续
        
        # 计算深度梯度
        depth_grad_x = F.conv2d(pred_depth, self.sobel_x, padding=1)
        depth_grad_y = F.conv2d(pred_depth, self.sobel_y, padding=1)
        
        # 计算RGB梯度（用于边缘检测）
        rgb_gray = rgb.mean(dim=1, keepdim=True)
        rgb_grad_x = F.conv2d(rgb_gray, self.sobel_x, padding=1)
        rgb_grad_y = F.conv2d(rgb_gray, self.sobel_y, padding=1)
        
        # 边缘权重（RGB边缘处权重低，允许深度不连续）
        edge_weight_x = torch.exp(-torch.abs(rgb_grad_x) * 10)
        edge_weight_y = torch.exp(-torch.abs(rgb_grad_y) * 10)
        
        # 加权平滑损失
        smooth_x = torch.abs(depth_grad_x) * edge_weight_x
        smooth_y = torch.abs(depth_grad_y) * edge_weight_y
        loss_smooth = (smooth_x.mean() + smooth_y.mean()) / 2
        
        # ========== 3. 不确定性正则化 ==========
        # 在稀疏点处，不确定性应该低（因为有真实观测）
        # 在其他位置，不确定性应该合理
        
        # 稀疏点处的不确定性应该接近0
        unc_at_sparse = pred_unc * valid_mask
        loss_unc_sparse = unc_at_sparse.sum() / num_valid
        
        # 其他位置的不确定性应该在合理范围内（防止过大或过小）
        unc_elsewhere = pred_unc * (1 - valid_mask)
        num_elsewhere = (1 - valid_mask).sum() + 1e-8
        
        # 鼓励不确定性在0.2-0.8之间（通过MSE到0.5）
        target_unc = torch.ones_like(unc_elsewhere) * 0.5
        loss_unc_reg = F.mse_loss(unc_elsewhere, target_unc, reduction='sum') / num_elsewhere
        
        loss_uncertainty = loss_unc_sparse + 0.1 * loss_unc_reg
        
        # ========== 总损失 ==========
        total_loss = (
            self.lambda_sparse * loss_sparse +
            self.lambda_smooth * loss_smooth +
            self.lambda_uncertainty * loss_uncertainty
        )
        
        return {
            'total': total_loss,
            'sparse': loss_sparse,
            'smooth': loss_smooth,
            'uncertainty': loss_uncertainty
        }


class R3LiveDataset(Dataset):
    """R3live稀疏深度数据集"""
    
    def __init__(self, data_dir, height=480, width=640, max_depth=100.0, augment=True):
        self.data_dir = Path(data_dir)
        self.height = height
        self.width = width
        self.max_depth = max_depth
        self.augment = augment
        
        self.rgb_files = sorted((self.data_dir / 'rgb').glob('*.png'))
        
        if len(self.rgb_files) == 0:
            raise ValueError(f"No images found in {self.data_dir / 'rgb'}")
        
        print(f"Found {len(self.rgb_files)} samples in {data_dir}")
    
    def __len__(self):
        return len(self.rgb_files)
    
    def __getitem__(self, idx):
        rgb_path = self.rgb_files[idx]
        stem = rgb_path.stem
        
        # 加载RGB
        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # 加载稀疏深度
        sparse_path = self.data_dir / 'sparse' / f'{stem}.png'
        sparse = cv2.imread(str(sparse_path), cv2.IMREAD_UNCHANGED)
        sparse = sparse.astype(np.float32) / 256.0
        
        # 调整尺寸
        rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        sparse = cv2.resize(sparse, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        
        # 数据增强
        if self.augment:
            if np.random.random() > 0.5:
                rgb = np.fliplr(rgb).copy()
                sparse = np.fliplr(sparse).copy()
            
            rgb = rgb.astype(np.float32)
            rgb *= np.random.uniform(0.9, 1.1, (1, 1, 3))
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        
        # 转Tensor
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        sparse = torch.from_numpy(sparse).unsqueeze(0).float()
        sparse = sparse.clamp(0, self.max_depth)
        
        return {'rgb': rgb, 'sparse_depth': sparse}


class Trainer:
    def __init__(self, model, train_loader, val_loader, lr, device, save_dir, use_amp):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        self.criterion = SparseDepthLoss().to(device)
        
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None
        
        self.epoch = 0
        self.best_loss = float('inf')
    
    def train_epoch(self):
        self.model.train()
        
        total_loss = 0
        total_sparse = 0
        total_smooth = 0
        total_unc = 0
        num_batches = 0
        
        for batch in self.train_loader:
            rgb = batch['rgb'].to(self.device)
            sparse = batch['sparse_depth'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    pred_depth, pred_unc = self.model(rgb, sparse)
                    losses = self.criterion(pred_depth, pred_unc, sparse, rgb)
                
                self.scaler.scale(losses['total']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred_depth, pred_unc = self.model(rgb, sparse)
                losses = self.criterion(pred_depth, pred_unc, sparse, rgb)
                losses['total'].backward()
                self.optimizer.step()
            
            total_loss += losses['total'].item()
            total_sparse += losses['sparse'].item()
            total_smooth += losses['smooth'].item()
            total_unc += losses['uncertainty'].item()
            num_batches += 1
        
        return {
            'total': total_loss / num_batches,
            'sparse': total_sparse / num_batches,
            'smooth': total_smooth / num_batches,
            'uncertainty': total_unc / num_batches
        }
    
    @torch.no_grad()
    def validate(self):
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        for batch in self.val_loader:
            rgb = batch['rgb'].to(self.device)
            sparse = batch['sparse_depth'].to(self.device)
            
            pred_depth, pred_unc = self.model(rgb, sparse)
            losses = self.criterion(pred_depth, pred_unc, sparse, rgb)
            
            # 计算MAE（只在稀疏点处）
            valid_mask = sparse > 0
            if valid_mask.sum() > 0:
                mae = torch.abs(pred_depth - sparse)[valid_mask].mean()
                total_mae += mae.item()
            
            total_loss += losses['total'].item()
            num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_mae': total_mae / num_batches
        }
    
    def save_checkpoint(self, filename, is_best=False):
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
    
    def train(self, num_epochs):
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"AMP: {self.use_amp}\n")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            train_losses = self.train_epoch()
            val_metrics = self.validate()
            
            self.scheduler.step()
            
            elapsed = time.time() - start_time
            lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Loss: {train_losses['total']:.4f} "
                  f"(Sparse: {train_losses['sparse']:.4f}, "
                  f"Smooth: {train_losses['smooth']:.4f}, "
                  f"Unc: {train_losses['uncertainty']:.4f}) "
                  f"LR: {lr:.6f} Time: {elapsed:.1f}s")
            
            if val_metrics:
                print(f"         Val Loss: {val_metrics['val_loss']:.4f}, "
                      f"MAE: {val_metrics['val_mae']:.3f}m")
            
            is_best = train_losses['total'] < self.best_loss
            if is_best:
                self.best_loss = train_losses['total']
                print(f"  ✓ Best model saved!")
            
            self.save_checkpoint('latest.pth', is_best=is_best)
            
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch+1}.pth')
        
        print(f"\nTraining completed! Best loss: {self.best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model-size', type=str, default='standard',
                       choices=['lite', 'standard', 'large'])
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--no-amp', action='store_true')
    
    args = parser.parse_args()
    
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("SGSNet Training - Sparse Depth Supervision")
    print("="*60)
    print(f"Data: {args.data_dir}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Device: {device}\n")
    
    # 创建模型
    if args.model_size == 'lite':
        model = SGSNetLite(max_depth=100.0)
    elif args.model_size == 'large':
        model = SGSNetLarge(max_depth=100.0)
    else:
        model = SGSNet(base_channels=32, max_depth=100.0)
    
    print(model.get_memory_footprint())
    
    # 数据集
    full_dataset = R3LiveDataset(
        args.data_dir, args.height, args.width, augment=True
    )
    
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples\n")
    
    # 训练
    trainer = Trainer(
        model, train_loader, val_loader,
        lr=args.lr, device=device,
        save_dir=args.save_dir,
        use_amp=not args.no_amp
    )
    
    trainer.train(args.epochs)


if __name__ == '__main__':
    main()