"""
SGSNet: Sparse-Guided Sparse-to-Dense Network for Depth Completion
适用于 Probabilistic Gaussian-LIC 项目的深度补全网络

特点:
1. 双输出: 稠密深度 + 不确定性图
2. 内存优化: 使用深度可分离卷积减少参数量
3. ONNX兼容: 避免使用不支持的操作
4. 实时性: 轻量级设计适合SLAM应用

作者: Probabilistic Gaussian-LIC Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积 - 减少参数量和计算量"""
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, 
                 padding: int = 1, bias: bool = True):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvBNReLU(nn.Module):
    """标准卷积块: Conv + BatchNorm + ReLU"""
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """残差块 - 增强特征学习能力"""
    def __init__(self, channels: int, use_separable: bool = True):
        super().__init__()
        if use_separable:
            self.conv1 = DepthwiseSeparableConv(channels, channels)
            self.conv2 = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels)
            )
        else:
            self.conv1 = ConvBNReLU(channels, channels)
            self.conv2 = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels)
            )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        return self.relu(out)


class SparseGuidanceModule(nn.Module):
    """
    稀疏引导模块 - 利用稀疏LiDAR点引导深度补全
    核心思想: 在有LiDAR观测的位置强制网络输出接近真实值
    """
    def __init__(self, channels: int):
        super().__init__()
        # 稀疏点特征提取
        self.sparse_conv = nn.Sequential(
            nn.Conv2d(1, channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # 有效性掩码
        self.mask_conv = nn.Sequential(
            nn.Conv2d(1, channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.Sigmoid()
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels + channels // 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features: torch.Tensor, sparse_depth: torch.Tensor) -> torch.Tensor:
        # 生成有效性掩码 (depth > 0 的位置)
        valid_mask = (sparse_depth > 0).float()
        
        # 提取稀疏深度特征
        sparse_features = self.sparse_conv(sparse_depth)
        
        # 生成注意力掩码
        attention = self.mask_conv(valid_mask)
        
        # 加权稀疏特征
        weighted_sparse = sparse_features * attention
        
        # 与主干特征融合
        fused = torch.cat([features, weighted_sparse], dim=1)
        return self.fusion(fused)


class EncoderBlock(nn.Module):
    """编码器块"""
    def __init__(self, in_channels: int, out_channels: int, 
                 num_residual: int = 2, use_separable: bool = True):
        super().__init__()
        # 下采样
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 残差块
        self.residuals = nn.Sequential(*[
            ResidualBlock(out_channels, use_separable) 
            for _ in range(num_residual)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.residuals(x)
        return x


class DecoderBlock(nn.Module):
    """解码器块 - 带跳跃连接"""
    def __init__(self, in_channels: int, skip_channels: int, 
                 out_channels: int, use_separable: bool = True):
        super().__init__()
        # 上采样使用转置卷积
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2, 
            kernel_size=2, stride=2, bias=False
        )
        self.bn_up = nn.BatchNorm2d(in_channels // 2)
        self.relu = nn.ReLU(inplace=True)
        
        # 跳跃连接融合
        total_channels = in_channels // 2 + skip_channels
        
        if use_separable:
            self.conv = nn.Sequential(
                DepthwiseSeparableConv(total_channels, out_channels),
                ResidualBlock(out_channels, use_separable)
            )
        else:
            self.conv = nn.Sequential(
                ConvBNReLU(total_channels, out_channels),
                ConvBNReLU(out_channels, out_channels)
            )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.relu(self.bn_up(x))
        
        # 处理尺寸不匹配 (ONNX兼容方式)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UncertaintyHead(nn.Module):
    """
    不确定性估计头
    输出范围 [0, 1]: 0=高置信度, 1=低置信度
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()  # 输出 [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DepthHead(nn.Module):
    """深度预测头"""
    def __init__(self, in_channels: int, max_depth: float = 100.0):
        super().__init__()
        self.max_depth = max_depth
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1),
            nn.Sigmoid()  # 归一化到 [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x * self.max_depth  # 缩放到实际深度范围


class SGSNet(nn.Module):
    """
    SGSNet: Sparse-Guided Sparse-to-Dense Network
    
    输入:
        - rgb: (B, 3, H, W) RGB图像, 值域 [0, 1]
        - sparse_depth: (B, 1, H, W) 稀疏深度图, 0表示无效
    
    输出:
        - dense_depth: (B, 1, H, W) 稠密深度图
        - uncertainty: (B, 1, H, W) 不确定性图, [0,1]
    
    参数:
        - base_channels: 基础通道数, 默认32 (轻量级)
        - max_depth: 最大深度值, 默认100m
        - use_separable: 是否使用深度可分离卷积节省内存
    """
    
    def __init__(self, 
                 base_channels: int = 32,
                 max_depth: float = 100.0,
                 use_separable: bool = True,
                 pretrained: bool = False):
        super().__init__()
        
        self.max_depth = max_depth
        self.use_separable = use_separable
        
        # 通道数配置 (金字塔结构)
        c1, c2, c3, c4, c5 = (
            base_channels,      # 32
            base_channels * 2,  # 64
            base_channels * 4,  # 128
            base_channels * 8,  # 256
            base_channels * 16  # 512
        )
        
        # ============ RGB编码器分支 ============
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, c1, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        
        self.rgb_enc1 = EncoderBlock(c1, c2, num_residual=2, use_separable=use_separable)
        self.rgb_enc2 = EncoderBlock(c2, c3, num_residual=2, use_separable=use_separable)
        self.rgb_enc3 = EncoderBlock(c3, c4, num_residual=3, use_separable=use_separable)
        self.rgb_enc4 = EncoderBlock(c4, c5, num_residual=3, use_separable=use_separable)
        
        # ============ 深度编码器分支 ============
        self.depth_stem = nn.Sequential(
            nn.Conv2d(1, c1 // 2, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1 // 2),
            nn.ReLU(inplace=True)
        )
        
        self.depth_enc1 = EncoderBlock(c1 // 2, c2 // 2, num_residual=1, use_separable=use_separable)
        self.depth_enc2 = EncoderBlock(c2 // 2, c3 // 2, num_residual=1, use_separable=use_separable)
        
        # ============ 特征融合模块 ============
        self.fusion1 = nn.Sequential(
            nn.Conv2d(c1 + c1 // 2, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        self.fusion2 = nn.Sequential(
            nn.Conv2d(c2 + c2 // 2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
        self.fusion3 = nn.Sequential(
            nn.Conv2d(c3 + c3 // 2, c3, 1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True)
        )
        
        # ============ 稀疏引导模块 ============
        self.sparse_guidance = SparseGuidanceModule(c5)
        
        # ============ 解码器 ============
        self.dec4 = DecoderBlock(c5, c4, c4, use_separable=use_separable)
        self.dec3 = DecoderBlock(c4, c3, c3, use_separable=use_separable)
        self.dec2 = DecoderBlock(c3, c2, c2, use_separable=use_separable)
        self.dec1 = DecoderBlock(c2, c1, c1, use_separable=use_separable)
        
        # ============ 最终上采样 ============
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        
        # ============ 输出头 ============
        self.depth_head = DepthHead(c1, max_depth=max_depth)
        self.uncertainty_head = UncertaintyHead(c1)
        
        # 权重初始化
        self._init_weights()
        
        if pretrained:
            self._load_pretrained()
    
    def _init_weights(self):
        """使用Kaiming初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _load_pretrained(self):
        """加载预训练权重 (如果可用)"""
        # 这里可以加载预训练的编码器权重
        pass
    
    def forward(self, rgb: torch.Tensor, sparse_depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            rgb: (B, 3, H, W) RGB图像
            sparse_depth: (B, 1, H, W) 稀疏深度图
        
        Returns:
            dense_depth: (B, 1, H, W) 稠密深度
            uncertainty: (B, 1, H, W) 不确定性图
        """
        # 保存原始尺寸用于最后的上采样
        original_size = rgb.shape[2:]
        
        # ============ RGB编码 ============
        rgb_s1 = self.rgb_stem(rgb)           # (B, c1, H/2, W/2)
        rgb_s2 = self.rgb_enc1(rgb_s1)        # (B, c2, H/4, W/4)
        rgb_s3 = self.rgb_enc2(rgb_s2)        # (B, c3, H/8, W/8)
        rgb_s4 = self.rgb_enc3(rgb_s3)        # (B, c4, H/16, W/16)
        rgb_s5 = self.rgb_enc4(rgb_s4)        # (B, c5, H/32, W/32)
        
        # ============ 深度编码 ============
        depth_s1 = self.depth_stem(sparse_depth)   # (B, c1/2, H/2, W/2)
        depth_s2 = self.depth_enc1(depth_s1)       # (B, c2/2, H/4, W/4)
        depth_s3 = self.depth_enc2(depth_s2)       # (B, c3/2, H/8, W/8)
        
        # ============ 多尺度特征融合 ============
        fused_s1 = self.fusion1(torch.cat([rgb_s1, depth_s1], dim=1))
        fused_s2 = self.fusion2(torch.cat([rgb_s2, depth_s2], dim=1))
        fused_s3 = self.fusion3(torch.cat([rgb_s3, depth_s3], dim=1))
        
        # ============ 稀疏引导 ============
        # 下采样稀疏深度到瓶颈尺寸
        sparse_bottleneck = F.interpolate(
            sparse_depth, 
            size=rgb_s5.shape[2:], 
            mode='nearest'
        )
        guided_features = self.sparse_guidance(rgb_s5, sparse_bottleneck)
        
        # ============ 解码 ============
        dec4 = self.dec4(guided_features, rgb_s4)  # (B, c4, H/16, W/16)
        dec3 = self.dec3(dec4, fused_s3)           # (B, c3, H/8, W/8)
        dec2 = self.dec2(dec3, fused_s2)           # (B, c2, H/4, W/4)
        dec1 = self.dec1(dec2, fused_s1)           # (B, c1, H/2, W/2)
        
        # ============ 最终上采样 ============
        features = self.final_upsample(dec1)       # (B, c1, H, W)
        
        # 确保尺寸匹配
        if features.shape[2:] != original_size:
            features = F.interpolate(features, size=original_size, mode='bilinear', align_corners=False)
        
        # ============ 输出 ============
        dense_depth = self.depth_head(features)
        uncertainty = self.uncertainty_head(features)
        
        # ============ 稀疏深度约束 ============
        # 在有LiDAR观测的位置，强制使用真实深度值，并将不确定性设为0
        # valid_mask = (sparse_depth > 0).float()
        # dense_depth = dense_depth * (1 - valid_mask) + sparse_depth * valid_mask
        # uncertainty = uncertainty * (1 - valid_mask)  # LiDAR点处不确定性为0
        
        # ============ 稀疏深度约束 ============
        # 训练时：让网络自由预测，用稀疏点验证
        # 推理时：可选择是否强制覆盖
        if not self.training:  # 只在eval模式强制覆盖
            valid_mask = (sparse_depth > 0).float()
            dense_depth = dense_depth * (1 - valid_mask) + sparse_depth * valid_mask
            uncertainty = uncertainty * (1 - valid_mask)
        
        return dense_depth, uncertainty
    
    def get_memory_footprint(self) -> str:
        """获取模型内存占用信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 估算内存 (假设float32)
        param_memory_mb = total_params * 4 / (1024 * 1024)
        
        return (
            f"Total parameters: {total_params:,}\n"
            f"Trainable parameters: {trainable_params:,}\n"
            f"Estimated parameter memory: {param_memory_mb:.2f} MB"
        )


class SGSNetLite(SGSNet):
    """
    轻量级SGSNet变体
    - 更少的通道数
    - 适合内存受限的设备
    """
    def __init__(self, max_depth: float = 100.0, pretrained: bool = False):
        super().__init__(
            base_channels=16,  # 减半
            max_depth=max_depth,
            use_separable=True,
            pretrained=pretrained
        )


class SGSNetLarge(SGSNet):
    """
    大型SGSNet变体
    - 更多通道数
    - 更高的精度
    """
    def __init__(self, max_depth: float = 100.0, pretrained: bool = False):
        super().__init__(
            base_channels=48,  # 增加
            max_depth=max_depth,
            use_separable=False,  # 使用标准卷积
            pretrained=pretrained
        )


# ============ ONNX导出辅助函数 ============

def create_sgsnet_for_export(
    model_size: str = 'standard',
    max_depth: float = 100.0
) -> SGSNet:
    """
    创建用于ONNX导出的SGSNet模型
    
    Args:
        model_size: 'lite', 'standard', 或 'large'
        max_depth: 最大深度值
    
    Returns:
        配置好的SGSNet模型
    """
    if model_size == 'lite':
        model = SGSNetLite(max_depth=max_depth)
    elif model_size == 'large':
        model = SGSNetLarge(max_depth=max_depth)
    else:
        model = SGSNet(max_depth=max_depth)
    
    return model


def verify_onnx_output(
    pytorch_model: SGSNet,
    onnx_path: str,
    height: int = 480,
    width: int = 640
) -> bool:
    """
    验证ONNX模型输出与PyTorch模型一致
    """
    import onnxruntime as ort
    import numpy as np
    
    # PyTorch推理
    pytorch_model.eval()
    with torch.no_grad():
        dummy_rgb = torch.randn(1, 3, height, width)
        dummy_depth = torch.randn(1, 1, height, width).abs() * 10
        dummy_depth[dummy_depth < 5] = 0  # 模拟稀疏深度
        
        if torch.cuda.is_available():
            pytorch_model = pytorch_model.cuda()
            dummy_rgb = dummy_rgb.cuda()
            dummy_depth = dummy_depth.cuda()
        
        pt_depth, pt_unc = pytorch_model(dummy_rgb, dummy_depth)
        pt_depth = pt_depth.cpu().numpy()
        pt_unc = pt_unc.cpu().numpy()
    
    # ONNX推理
    sess = ort.InferenceSession(onnx_path)
    ort_depth, ort_unc = sess.run(
        None,
        {
            'rgb': dummy_rgb.cpu().numpy(),
            'sparse_depth': dummy_depth.cpu().numpy()
        }
    )
    
    # 比较输出
    depth_diff = np.abs(pt_depth - ort_depth).max()
    unc_diff = np.abs(pt_unc - ort_unc).max()
    
    print(f"Max depth difference: {depth_diff:.6f}")
    print(f"Max uncertainty difference: {unc_diff:.6f}")
    
    # 允许一定的数值误差
    return depth_diff < 1e-4 and unc_diff < 1e-4


# ============ 测试代码 ============

if __name__ == "__main__":
    # 测试模型创建和前向传播
    print("=" * 60)
    print("SGSNet Model Test")
    print("=" * 60)
    
    # 创建模型
    model = SGSNet(base_channels=32, max_depth=100.0)
    print(model.get_memory_footprint())
    print()
    
    # 测试前向传播
    batch_size = 1
    height, width = 480, 640
    
    rgb = torch.randn(batch_size, 3, height, width)
    sparse_depth = torch.randn(batch_size, 1, height, width).abs() * 50
    sparse_depth[sparse_depth < 25] = 0  # 模拟稀疏深度 (~50% 覆盖率)
    
    print(f"Input RGB shape: {rgb.shape}")
    print(f"Input sparse depth shape: {sparse_depth.shape}")
    print(f"Sparse depth coverage: {(sparse_depth > 0).float().mean():.2%}")
    
    # GPU测试 (如果可用)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = model.to(device)
    rgb = rgb.to(device)
    sparse_depth = sparse_depth.to(device)
    
    # 预热
    with torch.no_grad():
        _ = model(rgb, sparse_depth)
    
    # 计时测试
    import time
    num_runs = 10
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            dense_depth, uncertainty = model(rgb, sparse_depth)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = (time.time() - start) / num_runs * 1000
    
    print(f"\nOutput dense depth shape: {dense_depth.shape}")
    print(f"Output uncertainty shape: {uncertainty.shape}")
    print(f"Dense depth range: [{dense_depth.min():.2f}, {dense_depth.max():.2f}]")
    print(f"Uncertainty range: [{uncertainty.min():.4f}, {uncertainty.max():.4f}]")
    print(f"Average inference time: {elapsed:.2f} ms")
    
    # 内存测试 (仅GPU)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(rgb, sparse_depth)
        peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Peak GPU memory: {peak_memory:.2f} MB")
    
    print("\n✓ SGSNet test passed!")