#!/usr/bin/env python3
import cv2
import numpy as np
import onnxruntime as ort
import os

models = [
    ('原始预训练', 'models/sgsnet.onnx'),
    ('训练后(带override)', 'models/sgsnet_trained_640x512.onnx'),
    ('训练后(无override)', 'models/sgsnet_trained_640x512_no_override.onnx'),
]

# 加载测试数据
rgb = cv2.imread('/root/sgsnet_training/sgsnet_data_640x512_resized/rgb/000000.png')
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
rgb = np.transpose(rgb, (2, 0, 1))[None, ...]

sparse = cv2.imread('/root/sgsnet_training/sgsnet_data_640x512_resized/sparse/000000.png', cv2.IMREAD_UNCHANGED)
sparse = sparse.astype(np.float32) / 256.0
sparse_input = sparse[None, None, ...]

print("="*60)
print("模型对比测试")
print("="*60)
print(f"输入sparse深度范围: [{sparse[sparse>0].min():.2f}, {sparse[sparse>0].max():.2f}]")
print()

for name, path in models:
    if not os.path.exists(path):
        print(f"❌ {name}: 文件不存在 ({path})")
        continue
    
    try:
        sess = ort.InferenceSession(path)
        dense, unc = sess.run(None, {'rgb': rgb, 'sparse_depth': sparse_input})
        
        valid = sparse > 0
        mae = np.abs(dense[0,0][valid] - sparse[valid]).mean()
        saturated = (dense > 95).sum()
        
        print(f"✅ {name}:")
        print(f"   输出范围: [{dense.min():.2f}, {dense.max():.2f}]")
        print(f"   稀疏点MAE: {mae:.3f}m")
        print(f"   饱和点(>95m): {saturated} ({saturated/dense.size*100:.2f}%)")
        print(f"   深度分布: <50m={((dense<50).sum()/dense.size*100):.1f}%, 50-95m={((dense>=50)&(dense<95)).sum()/dense.size*100:.1f}%, >95m={((dense>=95).sum()/dense.size*100):.1f}%")
        print()
    except Exception as e:
        print(f"❌ {name}: 加载失败 - {e}")
        print()
