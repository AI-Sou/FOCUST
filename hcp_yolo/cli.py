#!/usr/bin/env python3
"""
HCP-YOLO 统一命令行接口
整合所有功能的命令行工具
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Ensure this module is runnable as a standalone script from any working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Windows: avoid hard-crash when multiple OpenMP runtimes are loaded (common with torch/numpy/sklearn).
# Users can override explicitly by setting the env var themselves.
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_build(args):
    """构建数据集"""
    from hcp_yolo.dataset_builder import HCPDatasetBuilder

    print("=" * 60)
    print("HCP-YOLO 数据集构建")
    print("=" * 60)

    builder = HCPDatasetBuilder(
        anno_json=args.anno_json,
        images_dir=args.images_dir,
        output_dir=args.output,
        single_class=args.single_class,
        negative_ratio=args.negative_ratio,
        label_mode=args.label_mode,
        save_original_frames=not args.no_original_frames,
        save_hcp_full_images=not args.no_hcp_full_images,
        save_original_gt_visualizations=not args.no_original_gt_viz,
    )

    stats = builder.build()

    print(f"\n完成! 正样本: {stats['positive']}, 负样本: {stats['negative']}")


def cmd_train(args):
    """训练模型"""
    from hcp_yolo.trainer import HCPYOLOTrainer

    print("=" * 60)
    print("HCP-YOLO 模型训练")
    print("=" * 60)
    print(f"数据集: {args.dataset}")
    print(f"模型: {args.model}")
    print(f"设备: {args.device}")

    trainer = HCPYOLOTrainer(model_path=args.model, device=args.device)

    extra = {}
    # Optional ultralytics params (keep CLI backward-compatible)
    for key in ("imgsz", "workers", "lr0", "patience", "optimizer", "cache", "project", "name"):
        v = getattr(args, key, None)
        if v is not None:
            extra[key] = v

    model_path = trainer.train(
        dataset_path=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        **extra,
    )

    print(f"\n训练完成! 最佳模型: {model_path}")


def cmd_predict(args):
    """推理"""
    from hcp_yolo.inference import HCPYOLOInference

    print("=" * 60)
    print("HCP-YOLO 推理")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"输入: {args.input}")

    inferencer = HCPYOLOInference(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )

    # 单张推理
    if Path(args.input).is_file():
        result = inferencer.predict(args.input, use_sahi=args.sahi)

        print(f"\n检测结果: {result['num_detections']} 个目标")
        print(f"推理时间: {result['inference_time']:.4f}s")

        if args.output:
            import cv2
            vis = inferencer.visualize(result, args.input)
            cv2.imwrite(args.output, vis)
            print(f"可视化结果: {args.output}")

    # 目录推理
    elif Path(args.input).is_dir():
        results = inferencer.predict_directory(
            args.input,
            use_sahi=args.sahi,
            save_dir=Path(args.output) if args.output else None
        )

        print(f"\n完成! 处理了 {len(results)} 张图像")


def cmd_evaluate(args):
    """评估模型"""
    from hcp_yolo.evaluation import HCPYOLOEvaluator

    print("=" * 60)
    print("HCP-YOLO 模型评估")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"数据集: {args.dataset}")
    print(f"划分: {args.split}")

    evaluator = HCPYOLOEvaluator(args.model)

    metrics = evaluator.evaluate(
        dataset_path=args.dataset,
        split=args.split
    )

    print(f"\nmAP@0.5: {metrics.get('map50', 0):.4f}")
    print(f"mAP@0.5:0.95: {metrics.get('map50_95', 0):.4f}")
    print(f"Precision: {metrics.get('precision', 0):.4f}")
    print(f"Recall: {metrics.get('recall', 0):.4f}")
    print(f"F1-Score: {metrics.get('f1', 0):.4f}")


def cmd_full_pipeline(args):
    """完整流程"""
    from hcp_yolo.core import HCPYOLO

    print("=" * 60)
    print("HCP-YOLO 完整流程")
    print("=" * 60)

    yolo = HCPYOLO()

    result = yolo.build_train_evaluate(
        anno_json=args.anno_json,
        images_dir=args.images_dir,
        dataset_dir=args.dataset_output,
        model_path=args.model
    )

    print("\n" + "=" * 60)
    print("完整流程完成")
    print("=" * 60)
    print(f"正样本: {result['dataset_stats']['positive']}")
    print(f"最佳模型: {result['best_model']}")
    print(f"mAP@0.5: {result['metrics'].get('map50', 0):.4f}")


def cmd_explain_hcp(args):
    """输出 HCP / 数据集 / 评估逻辑说明"""
    print("=" * 60)
    print("HCP-YOLO：HCP 编码与数据集构建逻辑（说明）")
    print("=" * 60)
    print("核心代码：")
    print("  - hcp_yolo/hcp_encoder.py")
    print("  - hcp_yolo/dataset_builder.py")
    print("  - hcp_yolo/advanced_evaluation.py")
    print("")
    print("文档：")
    print("  - hcp_yolo/README.md")
    print("")
    print("关键点（当前默认行为）：")
    print("  - HCP：默认输出 first_appearance_map（HPYER Stage0-2 的 JET 时间图，背景置黑）")
    print("  - label_mode=last_frame：只用窗口最后一帧标注，避免重复框")
    print("  - 切片数据集会写 dataset_index.jsonl：支持把切片预测画回 original_images")


def main():
    parser = argparse.ArgumentParser(
        description="HCP-YOLO 统一命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 构建数据集
  python -m hcp_yolo build --anno-json annotations.json --images-dir ./images --output ./dataset

  # 训练模型
  python -m hcp_yolo train --dataset ./dataset --model yolo11n.pt --epochs 100

  # 推理
  python -m hcp_yolo predict --model best.pt --input image.jpg --output result.jpg

  # 评估
  python -m hcp_yolo evaluate --model best.pt --dataset ./dataset --split test

  # 完整流程
  python -m hcp_yolo full-pipeline --anno-json annotations.json --images-dir ./images
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 数据集构建命令
    build_parser = subparsers.add_parser('build', help='构建数据集')
    build_parser.add_argument('--anno-json', required=True, help='annotations.json路径')
    build_parser.add_argument('--images-dir', required=True, help='图像目录')
    build_parser.add_argument('--output', default='./hcp_dataset', help='输出目录')
    build_parser.add_argument('--single-class', action='store_true', help='单类别模式')
    build_parser.add_argument('--negative-ratio', type=float, default=0.3, help='负样本比例')
    build_parser.add_argument('--label-mode', default='last_frame', choices=['last_frame', 'all_frames'], help='标签来源策略')
    build_parser.add_argument('--no-original-frames', action='store_true', help='不保存 original_images')
    build_parser.add_argument('--no-hcp-full-images', action='store_true', help='不保存 hcp_full_images')
    build_parser.add_argument('--no-original-gt-viz', action='store_true', help='不保存原图GT可视化')
    build_parser.set_defaults(func=cmd_build)

    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--dataset', required=True, help='数据集路径')
    train_parser.add_argument('--model', default='yolo11n.pt', help='预训练模型')
    train_parser.add_argument('--output', default='./runs', help='输出目录')
    train_parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    train_parser.add_argument('--batch', type=int, default=4, help='批次大小')
    train_parser.add_argument('--device', default='auto', help='设备 (auto/cuda/cpu)')
    train_parser.add_argument('--imgsz', type=int, default=None, help='图像尺寸 (Ultralytics imgsz)')
    train_parser.add_argument('--workers', type=int, default=None, help='DataLoader workers')
    train_parser.add_argument('--lr0', type=float, default=None, help='初始学习率 (Ultralytics lr0)')
    train_parser.add_argument('--patience', type=int, default=None, help='Early stop patience')
    train_parser.add_argument('--optimizer', default=None, help='Optimizer (e.g. AdamW/SGD)')
    train_parser.add_argument('--cache', default=None, help='Cache mode (e.g. disk/ram/False)')
    train_parser.add_argument('--project', default=None, help='Ultralytics project dir')
    train_parser.add_argument('--name', default=None, help='Ultralytics run name')
    train_parser.set_defaults(func=cmd_train)

    # 推理命令
    predict_parser = subparsers.add_parser('predict', help='推理')
    predict_parser.add_argument('--model', default='best.pt', help='模型路径')
    predict_parser.add_argument('--input', required=True, help='输入图像或目录')
    predict_parser.add_argument('--output', help='输出路径')
    predict_parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    predict_parser.add_argument('--iou', type=float, default=0.45, help='IoU阈值')
    predict_parser.add_argument('--sahi', action='store_true', help='使用SAHI推理')
    predict_parser.set_defaults(func=cmd_predict)

    # 评估命令
    eval_parser = subparsers.add_parser('evaluate', help='评估模型')
    eval_parser.add_argument('--model', default='best.pt', help='模型路径')
    eval_parser.add_argument('--dataset', required=True, help='数据集路径')
    eval_parser.add_argument('--split', default='test', choices=['train', 'val', 'test'], help='数据集划分')
    eval_parser.set_defaults(func=cmd_evaluate)

    # 完整流程命令
    pipeline_parser = subparsers.add_parser('full-pipeline', help='完整流程 (构建->训练->评估)')
    pipeline_parser.add_argument('--anno-json', required=True, help='annotations.json路径')
    pipeline_parser.add_argument('--images-dir', required=True, help='图像目录')
    pipeline_parser.add_argument('--dataset-output', default='./hcp_dataset', help='数据集输出目录')
    pipeline_parser.add_argument('--model', default='yolo11n.pt', help='预训练模型')
    pipeline_parser.set_defaults(func=cmd_full_pipeline)

    # 说明命令
    explain_parser = subparsers.add_parser('explain-hcp', help='输出 HCP/数据集/评估逻辑说明')
    explain_parser.set_defaults(func=cmd_explain_hcp)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        args.func(args)
        return 0
    except Exception as e:
        logger.error(f"执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
