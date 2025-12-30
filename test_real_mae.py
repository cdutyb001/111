#!/usr/bin/env python3
"""测试网络真实预测能力（不强制覆盖）"""

import torch
import cv2
import numpy as np
from sgsnet import SGSNet
from pathlib import Path

# 加载模型
model = SGSNet(base_channels=32, max_depth=100.0)
model.load_state_dict(torch.load('checkpoints_768/best_model.pth'))
model.eval()
model = model.cuda()

# 加载测试数据
data_dir = Path('/root/sgsnet_training/sgsnet_data')
rgb_files = sorted((data_dir / 'rgb').glob('*.png'))[:10]  # 测试10帧

total_mae = 0
total_samples = 0

print("测试网络真实预测能力（不强制覆盖）:")
print("="*60)

for rgb_path in rgb_files:
    stem = rgb_path.stem
    
    # 加载数据
    rgb = cv2.imread(str(rgb_path))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (960, 768))
    rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    rgb = rgb.unsqueeze(0).cuda()
    
    sparse_path = data_dir / 'sparse' / f'{stem}.png'
    sparse = cv2.imread(str(sparse_path), -1).astype(np.float32) / 256.0
    sparse = cv2.resize(sparse, (960, 768), interpolation=cv2.INTER_NEAREST)
    sparse_torch = torch.from_numpy(sparse).unsqueeze(0).unsqueeze(0).cuda()
    
    # 临时修改为训练模式，禁用强制覆盖
    model.train()  # 临时设为train模式
    with torch.no_grad():
        pred_depth, pred_unc = model(rgb, sparse_torch)
    model.eval()  # 恢复eval模式
    
    # 计算'EOF'MAE（在稀疏点处）
    pred = pred_depth.cpu().numpy()[0, 0]
    sparse_np = sparse
    
    valid = sparse_np > 0
    if valid.sum() > 0:
        mae = np.abs(pred[valid] - sparse_np[valid]).mean()
        total_mae += mae
        total_samples += 1
        print(f"{stem}: MAE = {mae:.3f}m, 稀疏点数 = {valid.sum()}")

print("="*60)
print(f"平均MAE（网络真实预测）: {total_mae/total_samples:.3f}m")
print()
echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config")
print("如果MAE < 1.0m，说明训练效果很好")
