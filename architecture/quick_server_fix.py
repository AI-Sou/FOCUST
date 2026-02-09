#!/usr/bin/env python3
"""
服务器快速修复脚本 - 解决模块导入问题
在服务器上运行此脚本来修复导入问题
"""

import os
import sys
from pathlib import Path

def fix_train_py():
    """修复train.py的导入问题"""
    print("[FIX] Applying import fix to train.py...")

    # 找到train.py文件
    train_py_path = Path(__file__).parent / 'hcp_yolo' / 'train.py'

    if not train_py_path.exists():
        print(f"[ERROR] train.py not found at {train_py_path}")
        return False

    # 读取文件内容
    try:
        with open(train_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"[ERROR] Failed to read train.py: {e}")
        return False

    # 检查是否已经修复
    if "current_dir = os.path.dirname(os.path.abspath(__file__))" in content:
        print("[OK] train.py already contains import fix")
        return True

    # 应用修复 - 在文件开头添加路径设置
    import_fix = """
# 确保当前目录在Python路径中
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
"""

    # 在第一个import语句之前插入修复
    lines = content.split('\n')
    insert_index = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            insert_index = i
            break

    # 插入修复代码
    lines.insert(insert_index, import_fix)

    # 写回文件
    try:
        with open(train_py_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print("[OK] Import fix applied to train.py")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to write train.py: {e}")
        return False

def create_simple_import_test():
    """创建简单的导入测试"""
    print("[CREATE] Creating simple import test...")

    test_content = '''#!/usr/bin/env python3
"""
简单导入测试 - 验证修复是否生效
"""

import sys
import os

# 确保路径正确
current_dir = os.path.dirname(os.path.abspath(__file__))
hcp_yolo_dir = os.path.join(current_dir, 'hcp_yolo')
if hcp_yolo_dir not in sys.path:
    sys.path.insert(0, hcp_yolo_dir)

print("[TEST] Testing imports...")

try:
    # 测试内存优化处理器
    import memory_optimized_processor
    print("[OK] memory_optimized_processor imported")
except Exception as e:
    print(f"[ERROR] memory_optimized_processor failed: {e}")

try:
    # 测试自适应处理器
    from adaptive_multithread_processor import create_adaptive_processor
    print("[OK] adaptive_multithread_processor imported")
except Exception as e:
    print(f"[ERROR] adaptive_multithread_processor failed: {e}")

try:
    # 测试评估器
    from evaluation import HCPYOLOEvaluator
    print("[OK] evaluation imported")
except Exception as e:
    print(f"[ERROR] evaluation failed: {e}")

print("[DONE] Import test completed")
'''

    test_path = Path(__file__).parent / 'simple_import_test.py'
    try:
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        print(f"[OK] Created {test_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create test: {e}")
        return False

def main():
    """主修复函数"""
    print("[START] Quick Server Import Fix")
    print("=" * 50)

    # 修复train.py
    success1 = fix_train_py()

    # 创建测试文件
    success2 = create_simple_import_test()

    print("\n" + "=" * 50)
    if success1 and success2:
        print("[SUCCESS] Import fix completed!")
        print("\nNext steps:")
        print("1. Run: python simple_import_test.py")
        print("2. If tests pass, run your training script")
        print("3. For HCP-YOLO training (recommended CLI): python -m hcp_yolo train --dataset ./hcp_dataset --model model/yolo11n.pt --epochs 100 --batch 8")
        print("   (Config templates are under: hcp_yolo/configs/)")
    else:
        print("[FAILED] Some fixes failed")
        print("Please check the error messages above")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
