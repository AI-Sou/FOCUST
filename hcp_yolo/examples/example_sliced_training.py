#!/usr/bin/env python3
"""
HCP-YOLO 切片数据集构建示例

方案B：数据集预处理切片
- 训练和推理完全一致
- 最佳小目标检测性能
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from hcp_yolo.dataset_builder import HCPSlicingDatasetBuilder


def build_sliced_dataset_example():
    """
    切片数据集构建示例
    """
    print("=" * 70)
    print("HCP-YOLO 切片数据集构建示例（方案B）")
    print("=" * 70)

    # 配置路径
    anno_json = "path/to/annotations.json"
    images_dir = "path/to/images"
    output_dir = "./hcp_dataset_sliced"

    # 初始化切片构建器
    builder = HCPSlicingDatasetBuilder(
        anno_json=anno_json,
        images_dir=images_dir,
        output_dir=output_dir,

        # 基础配置
        single_class=True,          # 单类别：colony
        negative_ratio=0.3,          # 30%负样本

        # 切片配置（关键！）
        slice_size=640,              # 切片大小：640×640
        overlap_ratio=0.2,           # 重叠比例：20%

        # HCP配置
        hcp_config={
            'background_frames': 10,
            'encoding_mode': 'first_appearance_map',
            'anchor_channel': 'negative',
            'noise_sigma_multiplier': 1.0,
            'noise_min_std_level': 2.0,
            'bg_consistency_multiplier': 3.0,
            'bf_diameter': 9,
            'bf_sigmaColor': 75.0,
            'bf_sigmaSpace': 75.0,
            'temporal_consistency_enable': True,
            'temporal_consistency_frames': 2,
            'fog_suppression_enable': True,
            'fog_sigma_ratio': 0.02,
            'fog_sigma_cap': 80.0,
        }
    )

    # 构建数据集
    print("\n开始构建切片数据集...")
    print(f"输入: {images_dir}")
    print(f"输出: {output_dir}")
    print(f"切片大小: {builder.slice_size}×{builder.slice_size}")
    print(f"重叠比例: {builder.overlap_ratio}")

    stats = builder.build()

    print("\n" + "=" * 70)
    print("数据集构建完成！")
    print("=" * 70)
    print(f"正样本序列: {stats['positive']}")
    print(f"负样本序列: {stats['negative']}")
    print(f"总切片数: {stats['total_slices']}")
    print(f"平均每张图像切片数: {stats['avg_slices_per_image']:.1f}")
    print(f"输出目录: {output_dir}")
    print("=" * 70)


def train_with_sliced_dataset():
    """
    使用切片数据集训练
    """
    from hcp_yolo.trainer import HCPYOLOTrainer

    print("\n" + "=" * 70)
    print("使用切片数据集训练")
    print("=" * 70)

    trainer = HCPYOLOTrainer(
        model_path="yolo11x.pt",
        device="auto"
    )

    best_model = trainer.train(
        dataset_path="./hcp_dataset_sliced/dataset.yaml",
        output_dir="./runs/sliced_training",

        # 训练参数
        epochs=300,
        batch_size=16,              # 切片数据集可以用更大batch
        learning_rate=0.0001,
        patience=50
    )

    print(f"\n最佳模型: {best_model}")
    return best_model


def inference_comparison():
    """
    推理方式对比
    """
    from hcp_yolo.inference import HCPYOLOInference

    print("\n" + "=" * 70)
    print("推理方式对比")
    print("=" * 70)

    # 加载模型
    model_path = "runs/sliced_training/weights/best.pt"

    # 方式1: 标准推理（推荐）
    print("\n方式1: 标准推理（切片训练后推荐）")
    print("  - 4K HCP图像直接resize到640")
    print("  - 与训练数据一致（都是640）")
    print("  - 推理速度快")

    inferencer = HCPYOLOInference(model_path, conf_threshold=0.15)
    result = inferencer.predict("test_hcp_image.jpg", use_sahi=False)
    print(f"  检测到: {result['num_detections']} 个菌落")

    # 方式2: SAHI推理（可选）
    print("\n方式2: SAHI推理（可选，与训练完全一致）")
    print("  - 4K HCP图像切片到640")
    print("  - 与训练数据完全一致")
    print("  - 推理时间较长")

    result = inferencer.predict("test_hcp_image.jpg", use_sahi=True, slice_size=640)
    print(f"  检测到: {result['num_detections']} 个菌落")


def complete_workflow():
    """
    完整工作流程：数据集构建 → 训练 → 推理
    """
    print("=" * 70)
    print("方案B完整工作流程")
    print("=" * 70)

    print("\n步骤1: 构建切片数据集")
    print("-" * 70)
    print("""
from hcp_yolo.dataset_builder import HCPSlicingDatasetBuilder

builder = HCPSlicingDatasetBuilder(
    anno_json="annotations.json",
    images_dir="./images",
    output_dir="./hcp_dataset_sliced",
    single_class=True,
    negative_ratio=0.3,
    slice_size=640,          # 关键：切片大小
    overlap_ratio=0.2       # 关键：重叠比例
)
stats = builder.build()
    """)

    print("\n步骤2: 训练模型")
    print("-" * 70)
    print("""
from hcp_yolo.trainer import HCPYOLOTrainer

trainer = HCPYOLOTrainer(model_path="yolo11x.pt")
best_model = trainer.train(
    dataset_path="./hcp_dataset_sliced/dataset.yaml",
    epochs=300,
    batch_size=16,          # 切片数据集可以用更大batch
    learning_rate=0.0001
)
    """)

    print("\n步骤3: 推理")
    print("-" * 70)
    print("""
from hcp_yolo.inference import HCPYOLOInference

inferencer = HCPYOLOInference("best.pt")
result = inferencer.predict("test.jpg")
    """)

    print("\n" + "=" * 70)
    print("关键优势")
    print("=" * 70)
    print("""
✅ 训练和推理完全一致（都是640）
✅ 小目标保持原始大小（30px→30px，不是30px→5px）
✅ 训练速度快（batch_size可以更大）
✅ 内存需求低（~8GB vs ~16GB）
✅ 最佳小目标检测性能
    """)


def comparison_with_other_methods():
    """
    与其他方案对比
    """
    print("\n" + "=" * 70)
    print("方案对比")
    print("=" * 70)

    comparison = """
┌───────────────┬──────────────┬──────────────┬──────────────┐
│     特性      │   方案A      │   方案B      │   方案C      │
│               │ 1280训练     │  切片训练     │  640训练     │
├───────────────┼──────────────┼──────────────┼──────────────┤
│ input_size    │ 1280         │ 640          │ 640          │
│ batch_size    │ 2            │ 16           │ 16           │
│ 内存需求      │ ~16GB        │ ~8GB         │ ~8GB         │
│ 训练速度      │ 慢           │ 快           │ 快           │
│ 训练/推理一致 │ 部分匹配     │ ✅ 完全匹配  │ ❌ 不匹配     │
│ 小目标保留    │ 较好(10px)   │ ✅ 最好(30px)│ ❌ 差(5px)   │
│ 数据集大小    │ 正常         │ 大(36x)      │ 正常         │
│ 构建时间      │ 快           │ 慢           │ 快           │
├───────────────┼──────────────┼──────────────┼──────────────┤
│ 推荐度        │ ⭐⭐⭐        │ ⭐⭐⭐⭐⭐     │ ⭐⭐          │
└───────────────┴──────────────┴──────────────┴──────────────┘

方案A: 1280训练 + SAHI推理
  优点: 实现简单
  缺点: 训练慢，内存大，训练/推理不完全匹配

方案B: 切片训练（推荐）⭐
  优点: 训练快，内存小，完全一致，最佳性能
  缺点: 数据集构建时间长，磁盘占用大

方案C: 640训练 + SAHI推理
  优点: 训练快
  缺点: 训练/推理不匹配，小目标检测差
    """

    print(comparison)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HCP-YOLO 切片数据集构建示例")
    parser.add_argument("--mode", choices=["build", "train", "inference", "workflow", "compare"],
                       default="workflow", help="运行模式")

    args = parser.parse_args()

    if args.mode == "build":
        build_sliced_dataset_example()
    elif args.mode == "train":
        train_with_sliced_dataset()
    elif args.mode == "inference":
        inference_comparison()
    elif args.mode == "workflow":
        complete_workflow()
    elif args.mode == "compare":
        comparison_with_other_methods()
