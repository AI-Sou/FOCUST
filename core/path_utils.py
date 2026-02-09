# -*- coding: utf-8 -*-
"""
跨平台路径处理工具模块

提供统一的跨平台文件和路径操作工具，解决Windows、Linux、macOS之间的兼容性问题。

主要功能：
- 安全的路径规范化
- 跨平台的原子文件操作
- Unicode路径支持
- 路径验证和清理
"""

import os
import sys
import re
import tempfile
import shutil
from pathlib import Path
from typing import Union, Optional, List, Iterable


def normalize_path(path: Union[str, Path], resolve: bool = True) -> Path:
    """
    规范化路径，确保跨平台兼容性

    Args:
        path: 输入路径
        resolve: 是否解析为绝对路径

    Returns:
        规范化后的Path对象
    """
    path_obj = Path(path)

    if resolve:
        try:
            path_obj = path_obj.resolve()
        except (OSError, RuntimeError):
            # 某些情况下resolve可能失败（如路径不存在）
            path_obj = path_obj.absolute()

    return path_obj


def safe_path_join(*parts: Union[str, Path]) -> Path:
    """
    安全的路径连接，自动处理不同操作系统的分隔符

    Args:
        *parts: 路径组成部分

    Returns:
        连接后的Path对象
    """
    if not parts:
        return Path()

    # 转换所有部分为字符串并使用pathlib处理
    base = Path(parts[0])
    for part in parts[1:]:
        base = base / part

    return base


def ensure_dir_exists(path: Union[str, Path], parents: bool = True, exist_ok: bool = True) -> Path:
    """
    确保目录存在，如果不存在则创建

    Args:
        path: 目录路径
        parents: 是否创建父目录
        exist_ok: 如果目录已存在是否报错

    Returns:
        目录的Path对象
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=parents, exist_ok=exist_ok)
    return path_obj


def atomic_write(filepath: Union[str, Path], content: Union[str, bytes],
                 encoding: Optional[str] = 'utf-8', mode: str = 'w') -> bool:
    """
    原子性写入文件（先写临时文件，然后替换）

    这个方法在所有平台上都能保证原子性操作，避免文件损坏。

    Args:
        filepath: 目标文件路径
        content: 要写入的内容
        encoding: 文本编码（仅用于文本模式）
        mode: 写入模式 ('w' 文本, 'wb' 二进制)

    Returns:
        是否成功写入
    """
    filepath = Path(filepath)

    try:
        # 确保父目录存在
        ensure_dir_exists(filepath.parent)

        # 创建临时文件
        fd, temp_path = tempfile.mkstemp(
            dir=filepath.parent,
            prefix=f'.{filepath.name}.',
            suffix='.tmp'
        )

        try:
            # 写入内容到临时文件
            if 'b' in mode:
                os.write(fd, content if isinstance(content, bytes) else content.encode())
            else:
                os.write(fd, (content if isinstance(content, bytes) else content.encode(encoding)))

            os.close(fd)

            # 原子性替换（所有平台都支持）
            os.replace(temp_path, str(filepath))
            return True

        except Exception:
            # 清理临时文件
            try:
                os.close(fd)
            except:
                pass
            try:
                os.unlink(temp_path)
            except:
                pass
            raise

    except Exception as e:
        print(f"原子性写入文件失败 {filepath}: {e}")
        return False


def safe_remove(path: Union[str, Path], missing_ok: bool = True) -> bool:
    """
    安全删除文件或目录

    Args:
        path: 要删除的路径
        missing_ok: 如果文件不存在是否报错

    Returns:
        是否成功删除
    """
    path_obj = Path(path)

    try:
        if path_obj.is_file():
            path_obj.unlink(missing_ok=missing_ok)
        elif path_obj.is_dir():
            shutil.rmtree(path_obj)
        elif not missing_ok:
            raise FileNotFoundError(f"路径不存在: {path}")
        return True
    except Exception as e:
        if not missing_ok:
            raise
        print(f"删除失败 {path}: {e}")
        return False


def sanitize_filename(filename: str, replacement: str = '_', max_length: int = 255) -> str:
    """
    清理文件名，移除非法字符

    Args:
        filename: 原始文件名
        replacement: 替换非法字符的字符串
        max_length: 最大文件名长度

    Returns:
        清理后的文件名
    """
    # Windows和Unix共同的非法字符
    illegal_chars = r'[<>:"/\\|?*\x00-\x1f]'

    # 替换非法字符
    clean_name = re.sub(illegal_chars, replacement, filename)

    # 移除前后空格和点
    clean_name = clean_name.strip('. ')

    # 替换多个连续的替换字符
    clean_name = re.sub(f'{re.escape(replacement)}+', replacement, clean_name)

    # 限制长度
    if len(clean_name) > max_length:
        name, ext = os.path.splitext(clean_name)
        max_name_len = max_length - len(ext)
        clean_name = name[:max_name_len] + ext

    # 避免空文件名
    if not clean_name:
        clean_name = 'unnamed'

    return clean_name


def get_free_filename(directory: Union[str, Path], basename: str,
                      extension: str = '', max_attempts: int = 1000) -> Path:
    """
    获取一个不存在的文件名（如果文件存在，添加数字后缀）

    Args:
        directory: 目录路径
        basename: 基础文件名
        extension: 文件扩展名（包括点）
        max_attempts: 最大尝试次数

    Returns:
        可用的文件路径

    Raises:
        RuntimeError: 如果无法找到可用文件名
    """
    directory = Path(directory)
    basename = sanitize_filename(basename)

    # 尝试原始名称
    filepath = directory / f"{basename}{extension}"
    if not filepath.exists():
        return filepath

    # 添加数字后缀
    for i in range(1, max_attempts):
        filepath = directory / f"{basename}_{i}{extension}"
        if not filepath.exists():
            return filepath

    raise RuntimeError(f"无法在目录 {directory} 中找到可用的文件名")


def natural_sort_key(text: Union[str, Path]):
    """
    Key function for natural sorting (e.g. 2 < 10).

    Args:
        text: filename/path-like string.
    """
    parts = re.split(r"(\d+)", str(text))
    out = []
    for p in parts:
        if p.isdigit():
            try:
                out.append(int(p))
            except Exception:
                out.append(p)
        else:
            out.append(p.lower())
    return out


def collect_image_files(
    input_path: Union[str, Path],
    max_images: Optional[int] = None,
    recursive: bool = False,
    extensions: Optional[Iterable[str]] = None,
) -> List[Path]:
    """
    Collect image files from a folder / file / glob pattern.

    Args:
        input_path: folder, file, or glob pattern.
        max_images: optional cap on number of returned paths.
        recursive: when input is a folder, search recursively.
        extensions: allowed suffix set (case-insensitive), defaults to common image formats.
    """
    p = Path(input_path).expanduser()
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"} if extensions is None else {str(e).lower() for e in extensions}

    def _is_image(path: Path) -> bool:
        try:
            return path.is_file() and path.suffix.lower() in exts
        except Exception:
            return False

    files: List[Path] = []
    if p.exists() and p.is_dir():
        it = p.rglob("*") if recursive else p.iterdir()
        files = [x for x in it if _is_image(x)]
    elif p.exists() and p.is_file():
        files = [p]
    else:
        # Treat as glob under its parent.
        try:
            parent = p.parent if str(p.parent) else Path(".")
            pattern = p.name
            files = [x for x in parent.glob(pattern) if _is_image(x)]
        except Exception:
            files = []

    files = sorted(files, key=natural_sort_key)
    if max_images is not None:
        try:
            limit = int(max_images)
            if limit > 0:
                files = files[:limit]
        except Exception:
            pass
    return files


def is_safe_path(base_dir: Union[str, Path], path: Union[str, Path]) -> bool:
    """
    检查路径是否在指定的基础目录内（防止路径遍历攻击）

    Args:
        base_dir: 基础目录
        path: 要检查的路径

    Returns:
        路径是否安全
    """
    try:
        base_dir = Path(base_dir).resolve()
        path = Path(path).resolve()

        # 检查path是否是base_dir的子路径
        return path.is_relative_to(base_dir)
    except (ValueError, RuntimeError):
        return False


def get_size_human_readable(path: Union[str, Path]) -> str:
    """
    获取人类可读的文件/目录大小

    Args:
        path: 文件或目录路径

    Returns:
        格式化的大小字符串
    """
    path_obj = Path(path)

    if path_obj.is_file():
        size = path_obj.stat().st_size
    elif path_obj.is_dir():
        size = sum(f.stat().st_size for f in path_obj.rglob('*') if f.is_file())
    else:
        return "0 B"

    # 转换为合适的单位
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0

    return f"{size:.2f} PB"


def copy_with_progress(src: Union[str, Path], dst: Union[str, Path],
                       callback=None) -> bool:
    """
    复制文件，支持进度回调

    Args:
        src: 源文件路径
        dst: 目标文件路径
        callback: 进度回调函数 callback(bytes_copied, total_bytes)

    Returns:
        是否成功复制
    """
    src = Path(src)
    dst = Path(dst)

    try:
        # 确保目标目录存在
        ensure_dir_exists(dst.parent)

        total_size = src.stat().st_size
        bytes_copied = 0

        with open(src, 'rb') as fsrc:
            with open(dst, 'wb') as fdst:
                while True:
                    chunk = fsrc.read(64 * 1024)  # 64KB chunks
                    if not chunk:
                        break
                    fdst.write(chunk)
                    bytes_copied += len(chunk)

                    if callback:
                        callback(bytes_copied, total_size)

        return True

    except Exception as e:
        print(f"复制文件失败 {src} -> {dst}: {e}")
        return False


# 平台特定的路径分隔符
PATH_SEPARATOR = os.sep
IS_WINDOWS = sys.platform.startswith('win')
IS_POSIX = os.name == 'posix'


if __name__ == '__main__':
    # 测试代码
    print("跨平台路径工具模块测试")
    print(f"当前平台: {'Windows' if IS_WINDOWS else 'POSIX'}")
    print(f"路径分隔符: {PATH_SEPARATOR}")

    # 测试路径规范化
    test_path = normalize_path("./test/../example.txt")
    print(f"规范化路径: {test_path}")

    # 测试文件名清理
    dirty_name = "test<file>:name*.txt"
    clean_name = sanitize_filename(dirty_name)
    print(f"清理文件名: {dirty_name} -> {clean_name}")

    # 测试安全路径检查
    base = Path.cwd()
    safe = is_safe_path(base, base / "subdir" / "file.txt")
    unsafe = is_safe_path(base, base / ".." / ".." / "etc" / "passwd")
    print(f"安全路径检查: safe={safe}, unsafe={unsafe}")
