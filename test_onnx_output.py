#!/usr/bin/env python3
import cv2
import numpy as np
import onnxruntime as ort

print("加载ONNX模型...")
sess = ort.InferenceSession('models/sgsnet_trained_640x512.onnx')

print("加载测试数据...")
rgb = cv2.imread('/root/sgsnet_training/sgsnet_data_640x512_resized/rgb/000000.png')
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
rgb = np.transpose(rgb, (2, 0, 1))[None, ...]  # (1, 3, H, W)

sparse = cv2.imread('/root/sgsnet_training/sgsnet_data_640x512_resized/sparse/000000.png', cv2.IMREAD_UNCHANGED)
sparse = sparse.astype(np.float32) / 256.0
sparse = sparse[None, None, ...]  # (1, 1, H, W)

print(f"RGB shape: {rgb.shape}")
print(f"Sparse shape: {sparse.shape}")

print("ONNX推理...")
dense, unc = sess.run(None, {'rgb': rgb, 'sparse_depth': sparse})

print("="*60)
print("ONNX模型输出检查")
print("="*60)
print(f"输入sparse深度范围: [{sparse[sparse>0].min():.2f}, {sparse[sparse>0].max():.2f}]")
print(f"dense深度范围: [{dense.min():.2f}, {dense.max():.2f}]")
print(f"输出uncertainty范围: [{unc.min():.4f}, {unc.max():.4f}]")

# 在稀疏点处比较
valid = sparse[0,0] > 0
if valid.sum() > 0:
    mae = np.abs(dense[0,0][valid] - sparse[0,0][valid]).mean()
    print(f"稀疏点MAE: {mae:.3f}m")
    
    # 检查是否有异常输出
    saturated = (dense > 95).sum()
    print(f"饱和点(>95m): {saturated} ({saturated/dense.size*100:.2f}%)")
    
    # 统计深度分布
    print(f"\n深度分布:")
    print(f"  <10m: {(dense<10).sum()} ({(dense<10).sum()/dense.size*100:.1f}%)")
    print(f"  10-50m: {((dense>=10)&(dense<50)).sum()} ({((dense>=10)&(dense<50)).sum()/dense.size*100:.1f}%)")
    print(f"  50-95m: {((dense>=50)&(dense<95)).sum()} ({((dense>=50)&(dense<95)).sum()/dense.size*100:.1f}%)")
    print(f"  >95m: {(dense>=95).sum()} ({(dense>=95).sum()/dense.size*100:.1f}%)")

print("\n✓ 测试完成")
