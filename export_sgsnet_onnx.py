#!/usr/bin/env python3
"""
SGSNet ONNX导出脚本 - 简化版

使用方法:
    # 导出标准模型 (随机初始化)
    python3 export_sgsnet_onnx.py
    
    # 导出轻量级模型
    python3 export_sgsnet_onnx.py --model-size lite
    
    # 导出大型模型
    python3 export_sgsnet_onnx.py --model-size large
    
    # 指定输出路径和分辨率
    python3 export_sgsnet_onnx.py --output models/sgsnet.onnx --height 480 --width 640
    
    # 加载预训练权重
    python3 export_sgsnet_onnx.py --weights checkpoints/sgsnet.pth
"""

import torch
import torch.onnx
import argparse
import os
import sys

# 导入SGSNet模型
from sgsnet import SGSNet, SGSNetLite, SGSNetLarge


def export_onnx(
    weights_path: str = None,
    output_path: str = "models/sgsnet.onnx",
    height: int = 480,
    width: int = 640,
    model_size: str = 'standard',
    max_depth: float = 100.0,
    opset_version: int = 13
):
    """导出SGSNet为ONNX格式"""
    
    print("=" * 60)
    print("SGSNet ONNX Export")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # ========== 1. 创建模型 ==========
    print(f"\n[1/5] Creating {model_size} model...")
    
    if model_size == 'lite':
        model = SGSNetLite(max_depth=max_depth)
    elif model_size == 'large':
        model = SGSNetLarge(max_depth=max_depth)
    else:
        model = SGSNet(base_channels=32, max_depth=max_depth)
    
    # 加载预训练权重 (如果提供)
    if weights_path and os.path.exists(weights_path):
        print(f"      Loading weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print("      ✓ Weights loaded successfully")
    else:
        print("      Using randomly initialized weights")
    
    model.eval()
    print(f"\n{model.get_memory_footprint()}")
    
    # ========== 2. 选择设备 ==========
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[2/5] Using device: {device}")
    model = model.to(device)
    
    # ========== 3. 创建示例输入 ==========
    print(f"\n[3/5] Creating dummy inputs ({height}x{width})...")
    dummy_rgb = torch.randn(1, 3, height, width, device=device)
    dummy_sparse_depth = torch.randn(1, 1, height, width, device=device).abs() * 20
    dummy_sparse_depth[dummy_sparse_depth < 10] = 0  # 模拟稀疏深度
    
    # 测试前向传播
    with torch.no_grad():
        dense_depth, uncertainty = model(dummy_rgb, dummy_sparse_depth)
    print(f"      Output dense_depth shape: {dense_depth.shape}")
    print(f"      Output uncertainty shape: {uncertainty.shape}")
    
    # ========== 4. 导出ONNX ==========
    print(f"\n[4/5] Exporting to ONNX (opset {opset_version})...")
    
    # 动态轴配置
    dynamic_axes = {
        'rgb': {0: 'batch', 2: 'height', 3: 'width'},
        'sparse_depth': {0: 'batch', 2: 'height', 3: 'width'},
        'dense_depth': {0: 'batch', 2: 'height', 3: 'width'},
        'uncertainty': {0: 'batch', 2: 'height', 3: 'width'}
    }
    
    torch.onnx.export(
        model,
        (dummy_rgb, dummy_sparse_depth),
        output_path,
        input_names=['rgb', 'sparse_depth'],
        output_names=['dense_depth', 'uncertainty'],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        verbose=False
    )
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"      ✓ Exported to: {output_path}")
    print(f"      File size: {file_size_mb:.2f} MB")
    
    # ========== 5. 验证ONNX模型 ==========
    print("\n[5/5] Validating ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("      ✓ ONNX model validation passed!")
    except ImportError:
        print("      ⚠ onnx package not installed, skipping validation")
        print("      Install with: pip install onnx")
    except Exception as e:
        print(f"      ✗ Validation failed: {e}")
        return False
    
    # ========== 完成 ==========
    print("\n" + "=" * 60)
    print("✓ Export completed successfully!")
    print("=" * 60)
    
    # 打印使用提示
    print("\n📌 Usage in C++:")
    print(f"   // Load with ONNX Runtime or TensorRT")
    print(f"   // Input: rgb (1,3,{height},{width}), sparse_depth (1,1,{height},{width})")
    print(f"   // Output: dense_depth (1,1,{height},{width}), uncertainty (1,1,{height},{width})")
    
    print("\n📌 TensorRT conversion:")
    print(f"   trtexec --onnx={output_path} \\")
    print(f"           --saveEngine={output_path.replace('.onnx', '.trt')} \\")
    print(f"           --fp16")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Export SGSNet to ONNX format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 export_sgsnet_onnx.py                          # 导出标准模型
  python3 export_sgsnet_onnx.py --model-size lite        # 导出轻量级模型
  python3 export_sgsnet_onnx.py --weights model.pth      # 使用预训练权重
  python3 export_sgsnet_onnx.py --height 240 --width 320 # 指定分辨率
        """
    )
    
    parser.add_argument('--weights', '-w', type=str, default=None,
                        help='Path to pretrained weights (optional, uses random init if not provided)')
    parser.add_argument('--output', '-o', type=str, default='models/sgsnet.onnx',
                        help='Output ONNX file path (default: models/sgsnet.onnx)')
    parser.add_argument('--height', type=int, default=480,
                        help='Input image height (default: 480)')
    parser.add_argument('--width', type=int, default=640,
                        help='Input image width (default: 640)')
    parser.add_argument('--model-size', type=str, default='standard',
                        choices=['lite', 'standard', 'large'],
                        help='Model size: lite (~2MB), standard (~15MB), large (~50MB)')
    parser.add_argument('--max-depth', type=float, default=100.0,
                        help='Maximum depth value in meters (default: 100.0)')
    parser.add_argument('--opset', type=int, default=13,
                        help='ONNX opset version (default: 13)')
    
    args = parser.parse_args()
    
    success = export_onnx(
        weights_path=args.weights,
        output_path=args.output,
        height=args.height,
        width=args.width,
        model_size=args.model_size,
        max_depth=args.max_depth,
        opset_version=args.opset
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()