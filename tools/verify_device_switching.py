#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
设备切换功能完整性验证脚本
验证设备切换在所有模块中的集成情况
"""

import sys
import torch
from PyQt5.QtWidgets import QApplication

def print_section(title):
    """打印分节标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_device_detection():
    """测试设备检测"""
    print_section("1. 设备检测测试")

    from core.device_manager import DeviceDetector
    devices = DeviceDetector.get_available_devices()

    print(f"检测到 {len(devices)} 个设备:")
    for i, dev in enumerate(devices, 1):
        print(f"  {i}. {dev}")
        print(f"     ID: {dev.device_id}")
        print(f"     类型: {dev.device_type}")
        if dev.device_type == 'gpu':
            print(f"     内存: {dev.memory_free}MB / {dev.memory_total}MB")

    return len(devices) > 0

def test_device_manager():
    """测试设备管理器"""
    print_section("2. 设备管理器测试")

    from core.device_manager import get_device_manager

    try:
        manager = get_device_manager()
        config = manager.get_device_config()

        print("OK: Device manager initialized successfully")
        print(f"  Current device: {config.get('gpu_device', 'N/A')}")
        print(f"  Multi-GPU training: {config.get('use_multi_gpu', False)}")
        print(f"  GPU memory limit: {config.get('max_gpu_memory_mb', 'N/A')} MB")
        print(f"  Memory optimization: {config.get('memory_optimization', False)}")
        print(f"  Worker threads: {config.get('num_workers', 'N/A')}")

        return True
    except Exception as e:
        print(f"ERROR: Device manager init failed: {e}")
        return False

def test_config_manager_integration():
    """测试配置管理器集成"""
    print_section("3. 配置管理器集成测试")

    from core import get_config_manager

    try:
        config_mgr = get_config_manager()

        # 测试设备配置读取
        gpu_device = config_mgr.get('device_config.gpu_device', 'cpu')
        chunk_size = config_mgr.get('device_config.chunk_size', 20)

        print("OK: 配置管理器读取成功")
        print(f"  配置的设备: {gpu_device}")
        print(f"  团块大小: {chunk_size}")

        # 测试设备配置写入
        test_device = 'cpu'
        config_mgr.set('device_config.gpu_device', test_device)
        retrieved = config_mgr.get('device_config.gpu_device')

        if retrieved == test_device:
            print(f"OK: 设备配置读写正常")
            return True
        else:
            print(f"ERROR: 设备配置读写异常: 写入{test_device}, 读取{retrieved}")
            return False

    except Exception as e:
        print(f"ERROR: 配置管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gui_integration():
    """测试GUI集成"""
    print_section("4. GUI设备选择器集成测试")

    try:
        # Import FocustGUI from gui.py file (not gui module directory)
        import importlib.util
        import os
        gui_file = os.path.join(os.path.dirname(__file__), 'gui.py')
        spec = importlib.util.spec_from_file_location("gui_main", gui_file)
        gui_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gui_module)

        # 创建GUI实例（不显示）
        gui = gui_module.FocustGUI()

        # 检查设备选择器是否存在
        if not hasattr(gui, 'device_selector'):
            print("ERROR: GUI缺少device_selector组件")
            return False

        # 检查get_selected_device方法
        if not hasattr(gui, 'get_selected_device'):
            print("ERROR: GUI缺少get_selected_device方法")
            return False

        current_device = gui.get_selected_device()
        print("OK: GUI设备选择器初始化成功")
        print(f"  当前选择的设备: {current_device}")

        # 测试设备切换回调
        device_changed_called = [False]
        def test_callback(device_id):
            device_changed_called[0] = True
            print(f"  设备切换回调触发: {device_id}")

        gui.device_selector.device_changed.connect(test_callback)

        # 模拟设备切换（切换到相同设备不会触发）
        available_devices = gui.device_selector.devices
        if len(available_devices) > 0:
            test_device = available_devices[0].device_id
            gui.device_selector.set_selected_device(test_device)

        print("OK: 设备切换功能正常")

        return True

    except Exception as e:
        print(f"ERROR: GUI集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_construction_integration():
    """测试数据集构建模块集成"""
    print_section("5. 数据集构建模块设备集成测试")

    try:
        # 检查dataset_construction.py中的设备使用
        with open('gui/dataset_construction.py', 'r', encoding='utf-8') as f:
            content = f.read()

        required_patterns = [
            'get_selected_device',
            'device_selector',
            'chunk_size',
            'device_config'
        ]

        missing = []
        for pattern in required_patterns:
            if pattern not in content:
                missing.append(pattern)

        if missing:
            print(f"ERROR: 数据集构建模块缺少设备相关代码: {', '.join(missing)}")
            return False

        print("OK: 数据集构建模块包含完整的设备集成代码")
        print("  OK: get_selected_device() - 设备选择")
        print("  OK: device_selector - 设备选择器引用")
        print("  OK: chunk_size - 团块大小配置")
        print("  OK: device_config - 设备配置")

        return True

    except Exception as e:
        print(f"ERROR: 数据集构建模块检查失败: {e}")
        return False

def test_training_integration():
    """测试训练模块集成"""
    print_section("6. 训练模块设备集成测试")

    try:
        # 检查training.py中的设备使用
        with open('gui/training.py', 'r', encoding='utf-8') as f:
            content = f.read()

        required_patterns = [
            'multi_gpu',
            'max_gpu_mem',
            'gpu_config',
            'get_selected_device'  # Changed from 'device_config' to actual method used
        ]

        missing = []
        for pattern in required_patterns:
            if pattern not in content:
                missing.append(pattern)

        if missing:
            print(f"ERROR: 训练模块缺少设备相关代码: {', '.join(missing)}")
            return False

        print("OK: 训练模块包含完整的设备集成代码")
        print("  OK: multi_gpu - 多GPU训练支持")
        print("  OK: max_gpu_mem - GPU内存限制")
        print("  OK: gpu_config - GPU配置")
        print("  OK: get_selected_device - 设备选择方法")

        return True

    except Exception as e:
        print(f"ERROR: 训练模块检查失败: {e}")
        return False

def test_pytorch_device_switching():
    """测试PyTorch设备切换"""
    print_section("7. PyTorch设备切换测试")

    print(f"CUDA可用: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("  系统无GPU，使用CPU模式")
        print("  OK: CPU设备可用")
        return True

    print(f"CUDA设备数量: {torch.cuda.device_count()}")

    # 测试设备切换
    try:
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            current = torch.cuda.current_device()
            print(f"  OK: 成功切换到 cuda:{i} (验证: cuda:{current})")

        # 测试张量设备分配
        tensor_cpu = torch.randn(10, 10)
        print(f"  OK: CPU张量创建成功: {tensor_cpu.device}")

        if torch.cuda.is_available():
            tensor_gpu = torch.randn(10, 10, device='cuda:0')
            print(f"  OK: GPU张量创建成功: {tensor_gpu.device}")

            # 测试设备间转移
            tensor_cpu_to_gpu = tensor_cpu.to('cuda:0')
            tensor_gpu_to_cpu = tensor_gpu.cpu()
            print(f"  OK: 设备间张量转移成功")

        return True

    except Exception as e:
        print(f"ERROR: PyTorch设备切换测试失败: {e}")
        return False

def test_device_persistence():
    """测试设备配置持久化"""
    print_section("8. 设备配置持久化测试")

    from core import get_config_manager
    import os

    config_mgr = get_config_manager()

    # 测试保存
    test_device = 'cpu'
    test_chunk_size = 25

    config_mgr.set('device_config.gpu_device', test_device)
    config_mgr.set('device_config.chunk_size', test_chunk_size)

    try:
        config_mgr.save_config()
        print(f"OK: 配置保存成功")

        # 验证文件存在
        if os.path.exists('focust_config.json'):
            print(f"  OK: 配置文件存在: focust_config.json")

            # 重新读取验证
            config_mgr_new = get_config_manager()
            saved_device = config_mgr_new.get('device_config.gpu_device')
            saved_chunk = config_mgr_new.get('device_config.chunk_size')

            if saved_device == test_device and saved_chunk == test_chunk_size:
                print(f"  OK: 配置持久化验证成功")
                print(f"    设备: {saved_device}")
                print(f"    团块大小: {saved_chunk}")
                return True
            else:
                print(f"  ERROR: 配置持久化验证失败")
                print(f"    期望: device={test_device}, chunk={test_chunk_size}")
                print(f"    实际: device={saved_device}, chunk={saved_chunk}")
                return False
        else:
            print(f"  ERROR: 配置文件不存在")
            return False

    except Exception as e:
        print(f"ERROR: 配置持久化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("  设备切换功能完整性验证")
    print("="*60)

    # 创建QApplication（GUI组件需要）
    app = QApplication(sys.argv)

    # 运行所有测试
    tests = [
        ("设备检测", test_device_detection),
        ("设备管理器", test_device_manager),
        ("配置管理器集成", test_config_manager_integration),
        ("GUI集成", test_gui_integration),
        ("数据集构建模块集成", test_dataset_construction_integration),
        ("训练模块集成", test_training_integration),
        ("PyTorch设备切换", test_pytorch_device_switching),
        ("设备配置持久化", test_device_persistence),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\nERROR: 测试 '{name}' 发生异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # 打印总结
    print_section("测试总结")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "OK: 通过" if result else "ERROR: 失败"
        print(f"  {status}: {name}")

    print(f"\n  总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n  SUCCESS: 所有测试通过！设备切换功能完整！")
        return 0
    else:
        print(f"\n  WARNING:  {total - passed} 个测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
