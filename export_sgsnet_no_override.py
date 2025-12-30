#!/usr/bin/env python3
"""导出SGSNet ONNX模型（不带稀疏点强制覆盖）"""

import torch
import argparse
from sgsnet import SGSNet

def export_onnx_no_override(weights_path, output_path, height, width, model_size='standard'):
    print("="*60)
    print("SGSNet ONNX Export (No Override Mode)")
    print("="*60)
    
    # 创建模型
    print(f"\n[1/5] Creating {model_size} model...")
    if model_size == 'lite':
        from sgsnet import SGSNetLite
        model = SGSNetLite(max_depth=100.0)
    elif model_size == 'large':
        from sgsnet import SGSNetLarge
        model = SGSNetLarge(max_depth=100.0)
    else:
        model = SGSNet(base_channels=32, max_depth=100.0)
    
    # 加载权重
    print(f"      Loading weights from: {weights_path}")
    model.load_state_dict(torch.load(weights_path))
    print("      Success: Weights loaded")
    
    # 强制设为训练模式，禁用强制覆盖
    model.train()
    print("      Model set to TRAIN mode (disables sparse override)")
    
    # 关闭dropout/batchnorm的训练行为
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.BatchNorm2d)):
            m.eval()
    
    print(f"\n{model.get_memory_footprint()}\n")
    
    # 'ENDOFFILE'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[2/5] Using device: {device}")
    model = model.to(device)
    
    # 创建dummy输入
    print(f"\n[3/5] Creating dummy inputs ({height}x{width})...")
    dummy_rgb = torch.randn(1, 3, height, width).to(device)
    dummy_sparse = torch.randn(1, 1, height, width).abs().to(device) * 50
    dummy_sparse[dummy_sparse < 25] = 0
    
    with torch.no_grad():
        dense, unc = model(dummy_rgb, dummy_sparse)
        print(f"      Output dense_depth shape: {dense.shape}")
        print(f"      Output uncertainty shape: {unc.shape}")
    
    # 导出ONNX
    print(f"\n[4/5] Exporting to ONNX...")
    torch.onnx.export(
        model,
        (dummy_rgb, dummy_sparse),
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['rgb', 'sparse_depth'],
        output_names=['dense_depth', 'uncertainty'],
        dynamic_axes={
            'rgb': {0: 'batch_size'},
            'sparse_depth': {0: 'batch_size'},
            'dense_depth': {0: 'batch_size'},
            'uncertainty': {0: 'batch_size'}
        }
    )
    
    import os
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"      Exported to: {output_path}")
    print(f"      File size: {file_size:.2f} MB")
    
    # 验ONNX
    print(f"\n[5/5] Validating ONNX model...")
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("      ONNX model validation passed!")
    
    print("\n" + "="*60)
    print("Export completed successfully!")
    print("="*60)
    print("\nThis model does NOT force-override sparse points")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--model-size', default='standard', choices=['lite', 'standard', 'large'])
    
    args = parser.parse_args()
    export_onnx_no_override(args.weights, args.output, args.height, args.width, args.model_size)
