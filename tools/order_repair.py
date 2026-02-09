import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import os
import shutil
import threading
from pathlib import Path
from PIL import Image
import re

# 引入感知哈希库
import imagehash

# --- 核心函数：提高哈希算法的精度 ---
def calculate_hash(filepath, hash_size=16):
    """使用高精度感知哈希(dhash)计算图片的“视觉指纹”"""
    try:
        with Image.open(filepath) as img:
            return imagehash.dhash(img, hash_size=hash_size)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"无法处理图片 {filepath}: {e}")
        return None

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("数据集图片乱序修正工具 (v6.4 - 修正索引偏差)")
        self.root.geometry("800x650")

        self.replacement_records = []
        self.reference_index = [] # 用于存储预先计算的哈希索引

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill="both", expand=True)

        path_frame = ttk.LabelFrame(main_frame, text="文件夹路径设置", padding="10")
        path_frame.pack(fill="x", expand=True, pady=5)

        ttk.Label(path_frame, text="Images 文件夹:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.images_path = tk.StringVar()
        ttk.Entry(path_frame, textvariable=self.images_path, width=60).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(path_frame, text="浏览...", command=lambda: self.browse_folder(self.images_path)).grid(row=0, column=2, padx=5)

        ttk.Label(path_frame, text="Images2 文件夹:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.images2_path = tk.StringVar()
        ttk.Entry(path_frame, textvariable=self.images2_path, width=60).grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Button(path_frame, text="浏览...", command=lambda: self.browse_folder(self.images2_path)).grid(row=1, column=2, padx=5)

        ttk.Label(path_frame, text="参照文件夹根目录:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.ref_root_path = tk.StringVar()
        ttk.Entry(path_frame, textvariable=self.ref_root_path, width=60).grid(row=2, column=1, sticky="ew", padx=5)
        ttk.Button(path_frame, text="浏览...", command=lambda: self.browse_folder(self.ref_root_path)).grid(row=2, column=2, padx=5)

        path_frame.columnconfigure(1, weight=1)

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill="x", pady=5)

        self.debug_mode = tk.BooleanVar(value=True) # 默认开启调试
        debug_check = ttk.Checkbutton(control_frame, text="开启调试模式 (输出详细匹配日志)", variable=self.debug_mode)
        debug_check.pack(side="left", padx=5)
        
        self.run_button = ttk.Button(main_frame, text="开始检查与修正", command=self.start_processing_thread)
        self.run_button.pack(pady=10, fill="x")
        
        self.progress = ttk.Progressbar(main_frame, orient="horizontal", length=100, mode="determinate")
        self.progress.pack(pady=5, fill="x")

        log_frame = ttk.LabelFrame(main_frame, text="运行日志", padding="10")
        log_frame.pack(fill="both", expand=True, pady=5)
        self.log_area = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=15)
        self.log_area.pack(fill="both", expand=True)
        self.log_area.tag_configure("warning", foreground="orange")

    def browse_folder(self, string_var):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            string_var.set(folder_selected)

    def log(self, message, level="info"):
        self.root.after(0, self._log_thread_safe, message, level)

    def _log_thread_safe(self, message, level):
        if level == "warning":
            self.log_area.insert(tk.END, message + "\n", "warning")
        else:
            self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.root.update_idletasks()

    def set_progress(self, value, maximum=None):
        self.root.after(0, self._set_progress_thread_safe, value, maximum)

    def _set_progress_thread_safe(self, value, maximum):
        if maximum is not None:
            self.progress['maximum'] = maximum
        self.progress['value'] = value
        self.root.update_idletasks()

    def start_processing_thread(self):
        self.run_button.config(state="disabled")
        self.replacement_records = []
        self.log("="*50)
        self.log("任务启动 (v6.4 - 修正索引偏差)...")
        if self.debug_mode.get():
            self.log(">>> 调试模式已开启 <<<")
        thread = threading.Thread(target=self.run_correction_logic)
        thread.daemon = True
        thread.start()

    def find_max_numbered_file(self, folder_path, prefix, extension=".jpg"):
        max_num = -1
        max_file_path = None
        if not os.path.isdir(folder_path): return None, -1
        
        files = [f for f in os.listdir(folder_path) if f.startswith(prefix) and f.lower().endswith(extension.lower())]
        for f in files:
            try:
                name_part = Path(f).stem
                num_str = name_part.replace(prefix, "")
                num = int(num_str)
                if num > max_num:
                    max_num = num
                    max_file_path = os.path.join(folder_path, f)
            except (ValueError, TypeError): continue
        return max_file_path, max_num

    def find_max_numbered_ref_file(self, folder_path):
        max_num = -1
        max_filename = None
        if not os.path.isdir(folder_path): return -1, None

        numeric_files = [f for f in os.listdir(folder_path) if re.fullmatch(r'\d+\.jpg', f, re.IGNORECASE)]
        for f in numeric_files:
            try:
                num = int(Path(f).stem)
                if num > max_num:
                    max_num = num
                    max_filename = f
            except ValueError: continue
        return max_num, max_filename

    def build_reference_index(self, ref_root_dir):
        self.log(">>> [阶段1] 开始构建参照文件夹哈希索引...")
        self.reference_index = []
        
        all_dirs = [Path(dirpath) for dirpath, _, _ in os.walk(ref_root_dir)]
        self.set_progress(0, len(all_dirs))

        for i, dir_path in enumerate(all_dirs):
            self.set_progress(i + 1)
            max_num, front_filename = self.find_max_numbered_ref_file(dir_path)

            if max_num != -1 and front_filename:
                back_filename_pattern = f"{max_num}_back.jpg"
                back_filename_actual = None
                for f in os.listdir(dir_path):
                    if f.lower() == back_filename_pattern:
                        back_filename_actual = f
                        break
                
                if back_filename_actual:
                    front_img_path = dir_path / front_filename
                    back_img_path = dir_path / back_filename_actual
                    front_hash = calculate_hash(front_img_path)
                    back_hash = calculate_hash(back_img_path)

                    if front_hash and back_hash:
                        self.reference_index.append({
                            "folder_path": str(dir_path),
                            "max_num": max_num,
                            "front_hash": front_hash,
                            "back_hash": back_hash
                        })
                else:
                    self.log(f"[警告] 在 {dir_path} 中找到 '{front_filename}' 但未找到其对应的 _back 文件。此项将不会被索引。", level="warning")
        
        self.log(f">>> [阶段1] 索引构建完成！共找到 {len(self.reference_index)} 个有效的参照项。")

    def run_correction_logic(self):
        images_dir = self.images_path.get()
        images2_dir = self.images2_path.get()
        ref_root_dir = self.ref_root_path.get()

        if not all([images_dir, images2_dir, ref_root_dir]):
            self.log("错误: 请确保所有三个文件夹路径都已指定！")
            self.run_button.config(state="normal")
            return
        
        try:
            self.build_reference_index(ref_root_dir)
            if not self.reference_index:
                self.log("警告: 在参照目录中未能构建任何有效的哈希索引。请检查目录结构和文件。", level="warning")
        except Exception as e:
            self.log(f"!!! 构建索引时发生严重错误: {e}")
            self.run_button.config(state="normal")
            return
            
        self.log("\n>>> [阶段2] 开始匹配与修正任务...")
        try:
            subfolders = sorted([d for d in os.listdir(images_dir) if d.isdigit() and int(d) >= 400], key=int)
            if not subfolders:
                self.log(f"错误: 在 '{images_dir}' 中未找到起始序号为466或更大的数字文件夹。")
                self.run_button.config(state="normal")
                return
        except FileNotFoundError:
            self.log(f"错误: 找不到路径 '{images_dir}'")
            self.run_button.config(state="normal")
            return

        self.set_progress(0, len(subfolders))

        for i, folder_name in enumerate(subfolders):
            self.set_progress(i + 1)
            self.log(f"\n--- 正在处理文件夹: {folder_name} ---")

            current_images_subfolder = os.path.join(images_dir, folder_name)
            max_img1_path, max_num = self.find_max_numbered_file(current_images_subfolder, f"{folder_name}_000")
            if not max_img1_path: self.log(f"警告: 在 {current_images_subfolder} 中未找到文件，跳过。", level="warning"); continue
            hash1 = calculate_hash(max_img1_path)
            if not hash1: self.log(f"错误: 无法计算 {max_img1_path} 的哈希值，跳过。"); continue
            self.log(f"在 images/{folder_name} 找到最大序号图片 (序号 {max_num}, 共 {max_num} 张)")

            current_images2_subfolder = os.path.join(images2_dir, folder_name)
            max_img2_path, _ = self.find_max_numbered_file(current_images2_subfolder, f"{folder_name}_000")
            if not max_img2_path: self.log(f"错误: 在 images2/{folder_name} 中未找到对应文件，跳过。"); continue
            hash2_key = calculate_hash(max_img2_path)
            if not hash2_key: self.log(f"错误: 无法计算 {max_img2_path} 的哈希值，跳过。"); continue
            self.log(f"得到待匹配的'钥匙'哈希 (来自 images2/{folder_name})")

            best_match = None
            min_distance = float('inf') 

            for ref_item in self.reference_index:
                # ==========================================================
                # 关键修复 1：匹配参照源的最大序号 (N-1)
                # ==========================================================
                if ref_item["max_num"] != max_num - 1:
                    continue

                hash3_lock = ref_item["front_hash"]
                distance = hash2_key - hash3_lock
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = ref_item
                
                if self.debug_mode.get():
                    self.log(f"  [调试] 对比参照项 '{Path(ref_item['folder_path']).name}' (其max_num为{ref_item['max_num']})... 距离为: {distance}")

            if best_match:
                self.log(f"      >>> 搜索完成。找到的最佳匹配项是 '{Path(best_match['folder_path']).name}'，最小距离为: {min_distance}")
                
                hash4_verify = best_match["back_hash"]
                self.log(f"  -> 从最佳匹配项的 back.jpg 计算得到哈希4: {hash4_verify}")
                self.log(f"  -> 待验证的原始哈希1是: {hash1}")

                if hash4_verify == hash1:
                    self.log("  -> 检查通过: 哈希4 == 哈希1。此文件夹图片序列正确。")
                else:
                    self.log("  >>> 检查发现问题: 哈希4 != 哈希1。需要修正！")
                    self.log("  >>> 开始使用最佳匹配源的 _back.jpg 系列文件执行替换操作...")
                    
                    files_replaced_count = 0
                    try:
                        correct_ref_subfolder_path = best_match["folder_path"]
                        source_files_list = {f.lower(): f for f in os.listdir(correct_ref_subfolder_path)}
                        
                        for num_order in range(1, max_num + 1):
                            dest_filename = f"{folder_name}_000{num_order:02d}.jpg"
                            dest_file = Path(current_images_subfolder) / dest_filename
                            
                            # ==========================================================
                            # 关键修复 2：源文件序号从 0 开始 (num_order - 1)
                            # ==========================================================
                            source_back_pattern = f"{num_order - 1}_back.jpg"
                            if source_back_pattern in source_files_list:
                                source_file_actual_name = source_files_list[source_back_pattern]
                                source_file = Path(correct_ref_subfolder_path) / source_file_actual_name
                                shutil.copy2(source_file, dest_file)
                                files_replaced_count += 1
                        self.log(f"  >>> 文件夹 {folder_name} 修正完成，共替换 {files_replaced_count} 个文件。")
                        
                        record = { "target": folder_name, "source": correct_ref_subfolder_path, "count": files_replaced_count }
                        self.replacement_records.append(record)

                    except Exception as e:
                        self.log(f"!!! 替换过程中发生严重错误: {e}")
            else:
                self.log(f"!!! 验证失败: 索引中没有找到任何与 '最大序号为 {max_num - 1}' 相匹配的参照项。", level="warning")

        self.log("\n" + "="*50)
        self.log("所有任务处理完毕！")
        self.run_button.config(state="normal")
        
        if self.replacement_records:
            self.log("检测到有替换操作，正在生成总结报告...")
            self.root.after(100, self.show_summary_report)

    def show_summary_report(self):
        report_window = tk.Toplevel(self.root)
        report_window.title("替换操作总结报告")
        report_window.geometry("900x400")

        frame = ttk.Frame(report_window, padding="10")
        frame.pack(expand=True, fill="both")

        cols = ("#", "被修正的文件夹", "修正源文件夹", "替换文件数")
        tree = ttk.Treeview(frame, columns=cols, show="headings")
        
        for col_name in cols: tree.heading(col_name, text=col_name)
        
        tree.column("#", width=50, anchor="center")
        tree.column("被修正的文件夹", width=150, anchor="w")
        tree.column("修正源文件夹", width=500, anchor="w")
        tree.column("替换文件数", width=100, anchor="center")

        for i, record in enumerate(self.replacement_records, start=1):
            tree.insert("", "end", values=(i, record["target"], record["source"], record["count"]))

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        close_button = ttk.Button(report_window, text="关闭报告", command=report_window.destroy)
        close_button.pack(pady=10)


if __name__ == "__main__":
    # 确保运行代码前已安装必要的库:
    # pip install Pillow imagehash
    root = tk.Tk()
    app = App(root)
    root.mainloop()