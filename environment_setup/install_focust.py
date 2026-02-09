#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FOCUST 跨平台智能安装启动器
FOCUST Cross-Platform Intelligent Installation Launcher

自动检测操作系统并执行相应的安装脚本
Automatically detects OS and executes appropriate installation script
"""

import os
import sys
import platform
import subprocess
import shutil
import time
import urllib.request
from pathlib import Path

def print_banner():
    """打印启动横幅"""
    print("=" * 60)
    print("    FOCUST 跨平台智能安装启动器")
    print("    FOCUST Cross-Platform Installation Launcher")
    print("=" * 60)
    print()

def detect_platform():
    """检测操作系统平台"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "windows":
        return "windows", "x64" if machine in ["amd64", "x86_64"] else "x86"
    elif system == "darwin":
        return "macos", "arm64" if machine == "arm64" else "x64"
    elif system == "linux":
        return "linux", machine
    else:
        return "unknown", machine

def detect_gpu_capabilities():
    """检测GPU类型和能力"""
    gpu_info = {
        "type": "cpu",
        "details": "CPU模式",
        "driver_version": None,
        "compute_capability": None
    }
    
    system = platform.system().lower()
    
    try:
        if system == "windows" or system == "linux":
            # 检测NVIDIA GPU
            import subprocess
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader,nounits"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                if lines:
                    parts = lines[0].split(', ')
                    if len(parts) >= 3:
                        gpu_info["type"] = "cuda"
                        gpu_info["details"] = f"NVIDIA {parts[0]}"
                        gpu_info["driver_version"] = parts[1]
                        gpu_info["compute_capability"] = parts[2]
        elif system == "darwin":
            # macOS Metal检测
            machine = platform.machine().lower()
            if machine == "arm64":
                gpu_info["type"] = "mps"
                gpu_info["details"] = "Apple Silicon (MPS支持)"
            else:
                gpu_info["details"] = "Intel Mac (CPU模式)"
    except Exception as e:
        print(f"GPU检测失败: {e}")
    
    return gpu_info

def check_prerequisites():
    """检查先决条件"""
    print("🔍 检查先决条件...")
    issues = []
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        issues.append(f"Python版本过低: {python_version.major}.{python_version.minor}.{python_version.micro}，需要Python 3.8+")
    else:
        print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查conda
    conda_path = shutil.which("conda")
    if not conda_path:
        issues.append("未找到conda，请先安装Miniconda或Anaconda")
        issues.append("   下载地址: https://docs.conda.io/en/latest/miniconda.html")
    else:
        print(f"✅ Conda已安装: {conda_path}")
        
        # 检查conda版本
        try:
            result = subprocess.run(["conda", "--version"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ Conda版本: {result.stdout.strip()}")
            else:
                issues.append("Conda命令执行失败")
        except Exception as e:
            issues.append(f"Conda版本检查失败: {e}")
    
    # 检查磁盘空间 (需要至少5GB)
    try:
        import shutil
        free_space = shutil.disk_usage(Path.home()).free / (1024**3)  # GB
        if free_space < 5:
            issues.append(f"磁盘空间不足: {free_space:.1f}GB可用，建议至少5GB")
        else:
            print(f"✅ 磁盘空间: {free_space:.1f}GB可用")
    except Exception as e:
        issues.append(f"磁盘空间检查失败: {e}")
    
    # 检查网络连接
    try:
        import urllib.request
        urllib.request.urlopen('https://conda.anaconda.org', timeout=10)
        print("✅ 网络连接正常")
    except Exception:
        issues.append("网络连接失败，可能影响包下载")
    
    if issues:
        print("\n❌ 发现以下问题:")
        for issue in issues:
            print(f"   • {issue}")
        return False
    
    return True

def run_installation_script(platform_name, architecture, gpu_info):
    """运行相应的安装脚本"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # 设置环境变量传递GPU信息
    env = os.environ.copy()
    env['FOCUST_GPU_TYPE'] = gpu_info['type']
    env['FOCUST_GPU_DETAILS'] = gpu_info['details']
    
    try:
        if platform_name == "windows":
            script_path = script_dir / "setup_focust_env_improved.bat"
            if not script_path.exists():
                script_path = script_dir / "setup_focust_env.bat"
            
            if script_path.exists():
                print(f"🚀 启动Windows安装脚本: {script_path}")
                print(f"📁 项目根目录: {project_root}")
                
                # 使用subprocess.run with better error handling
                result = subprocess.run([
                    "cmd", "/c", str(script_path)
                ], cwd=project_root, env=env, shell=False, 
                   capture_output=False, text=True, timeout=3600)  # 1小时超时
                
                return result.returncode == 0
            else:
                print("❌ 未找到Windows安装脚本")
                return False
                
        elif platform_name in ["macos", "linux"]:
            script_path = script_dir / "setup_focust_env_improved.sh"
            if not script_path.exists():
                script_path = script_dir / "setup_focust_env.sh"
            
            if script_path.exists():
                print(f"🚀 启动{platform_name.upper()}安装脚本: {script_path}")
                print(f"📁 项目根目录: {project_root}")
                
                # 确保脚本有执行权限
                os.chmod(script_path, 0o755)
                
                # 使用subprocess.run with better error handling
                result = subprocess.run([
                    "/bin/bash", str(script_path)
                ], cwd=project_root, env=env,
                   capture_output=False, text=True, timeout=3600)  # 1小时超时
                
                return result.returncode == 0
            else:
                print(f"❌ 未找到{platform_name.upper()}安装脚本")
                return False
        else:
            print(f"❌ 不支持的平台: {platform_name}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 安装脚本执行超时 (>1小时)")
        return False
    except Exception as e:
        print(f"❌ 执行安装脚本时出错: {e}")
        return False

def show_platform_specific_notes(platform_name, architecture):
    """显示平台特定的注意事项"""
    print("\n📋 平台特定注意事项:")
    
    if platform_name == "windows":
        print("• Windows平台注意事项:")
        print("  - 建议以管理员权限运行")
        print("  - 确保Windows Defender不会干扰安装")
        print("  - 支持NVIDIA GPU CUDA加速")
        
    elif platform_name == "macos":
        print("• macOS平台注意事项:")
        print("  - 可能需要安装Xcode Command Line Tools")
        print("  - Apple Silicon (M1/M2)支持MPS加速")
        print("  - Intel Mac使用CPU模式")
        if architecture == "arm64":
            print("  - 检测到Apple Silicon，将使用优化的配置")
        
    elif platform_name == "linux":
        print("• Linux平台注意事项:")
        print("  - 支持NVIDIA GPU CUDA加速")
        print("  - 可能需要安装额外的系统依赖")
        print("  - 建议使用Ubuntu 18.04+或CentOS 7+")

def main():
    """主函数"""
    print_banner()
    
    # 检测平台
    platform_name, architecture = detect_platform()
    print(f"🖥️  检测到平台: {platform_name} ({architecture})")
    
    # 检测GPU
    gpu_info = detect_gpu_capabilities()
    print(f"🎮 GPU信息: {gpu_info['details']}")
    if gpu_info.get('driver_version'):
        print(f"   驱动版本: {gpu_info['driver_version']}")
    if gpu_info.get('compute_capability'):
        print(f"   计算能力: {gpu_info['compute_capability']}")
    
    # 显示平台特定注意事项
    show_platform_specific_notes(platform_name, architecture)
    print()
    
    # 检查先决条件
    if not check_prerequisites():
        print("\n❌ 先决条件检查失败，请解决上述问题后重新运行")
        print("\n常见解决方案:")
        print("1. 更新Python: https://www.python.org/downloads/")
        print("2. 安装Conda: https://docs.conda.io/en/latest/miniconda.html")
        print("3. 检查网络连接和防火墙设置")
        print("4. 确保有足够的磁盘空间 (推荐>5GB)")
        sys.exit(1)
    
    print()
    
    # 显示安装概要
    print("📋 安装概要:")
    print(f"   平台: {platform_name} ({architecture})")
    print(f"   GPU: {gpu_info['type']} - {gpu_info['details']}")
    print(f"   预计安装时间: 10-30分钟 (取决于网络速度)")
    print(f"   磁盘空间需求: ~3-5GB")
    print()
    
    # 确认安装
    try:
        response = input("🤔 是否继续安装Focust环境? (y/N): ").strip().lower()
        if response not in ['y', 'yes', '是', '确定']:
            print("安装已取消")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\n安装已取消")
        sys.exit(0)
    
    print()
    print("🚀 开始安装，请耐心等待...")
    print("💡 提示: 首次安装可能需要较长时间下载依赖包")
    print()
    
    # 运行安装脚本
    start_time = time.time()
    success = run_installation_script(platform_name, architecture, gpu_info)
    end_time = time.time()
    
    print(f"\n⏱️  安装耗时: {end_time - start_time:.1f}秒")
    
    if success:
        print("\n✅ 安装完成！")
        print("\n📖 使用方法:")
        print("1. conda activate focust")
        print("2. python gui.py")
        print("\n🔧 故障排除:")
        print("如遇问题请查看: environment_setup/ENVIRONMENT_SETUP.md")
    else:
        print("\n❌ 安装失败")
        print("\n🔧 故障排除建议:")
        print("1. 检查网络连接是否稳定")
        print("2. 尝试更换conda镜像源")
        print("3. 清理conda缓存: conda clean --all")
        print("4. 查看详细错误日志")
        print("5. 参考文档: environment_setup/ENVIRONMENT_SETUP.md")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断安装")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 安装过程中出现错误: {e}")
        sys.exit(1)
