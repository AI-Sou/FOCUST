import os
import sys
import shutil
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QListWidget, QListWidgetItem, QPushButton, QFileDialog, QLabel,
                            QScrollArea, QInputDialog, QMessageBox, QListView, QTreeView)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("文件夹图片可视化工具")
        self.setGeometry(100, 100, 1200, 800)
        
        # 存储当前选择的大文件夹
        self.selected_folders = []
        # 存储所有子文件夹的路径
        self.subfolders = []
        # 当前显示的图片路径
        self.current_image_path = ""
        # 新建文件夹路径
        self.new_folder_path = ""
        
        # 创建主窗口布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # 创建左侧控制面板
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        
        # 创建选择文件夹按钮
        self.select_folder_btn = QPushButton("选择大文件夹")
        self.select_folder_btn.clicked.connect(self.select_folders)
        self.left_layout.addWidget(self.select_folder_btn)
        
        # 创建子文件夹列表
        self.subfolder_list = QListWidget()
        self.subfolder_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.subfolder_list.itemClicked.connect(self.display_image)
        self.left_layout.addWidget(QLabel("子文件夹列表:"))
        self.left_layout.addWidget(self.subfolder_list)
        
        # 创建新建文件夹按钮
        self.create_folder_btn = QPushButton("创建新文件夹")
        self.create_folder_btn.clicked.connect(self.create_new_folder)
        self.left_layout.addWidget(self.create_folder_btn)
        
        # 创建移动文件夹按钮
        self.move_folders_btn = QPushButton("移动选中文件夹到新文件夹")
        self.move_folders_btn.clicked.connect(self.move_folders)
        self.left_layout.addWidget(self.move_folders_btn)
        
        # 添加左侧面板到主布局
        self.main_layout.addWidget(self.left_panel, 1)
        
        # 创建右侧图片展示区
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        
        # 图片信息标签
        self.image_info_label = QLabel("未选择图片")
        self.right_layout.addWidget(self.image_info_label)
        
        # 创建滚动区域用于显示图片
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        self.right_layout.addWidget(self.scroll_area)
        
        # 添加右侧面板到主布局
        self.main_layout.addWidget(self.right_panel, 3)
        
    def select_folders(self):
        """选择多个大文件夹"""
        dialog = QFileDialog(self, "选择大文件夹")
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.DontResolveSymlinks, True)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        
        # 设置文件视图为多选模式
        listview = dialog.findChild(QListView, "listView")
        if listview:
            listview.setSelectionMode(QListView.ExtendedSelection)
        treeview = dialog.findChild(QTreeView)
        if treeview:
            treeview.setSelectionMode(QTreeView.ExtendedSelection)
        
        # 如果用户点击了确定
        if dialog.exec_():
            paths = dialog.selectedFiles()
            if paths:
                self.selected_folders = paths
                self.populate_subfolder_list()
    
    def populate_subfolder_list(self):
        """填充子文件夹列表"""
        self.subfolder_list.clear()
        self.subfolders = []
        
        for folder in self.selected_folders:
            # 获取该大文件夹下的所有子文件夹
            try:
                subdirs = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
                for subdir in subdirs:
                    full_path = os.path.join(folder, subdir)
                    self.subfolders.append(full_path)
                    self.subfolder_list.addItem(f"{os.path.basename(folder)}/{subdir}")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"读取文件夹错误: {str(e)}")
        
        if not self.subfolders:
            QMessageBox.information(self, "提示", "未找到子文件夹")
    
    def display_image(self, item):
        """显示选中子文件夹中的XXX_back后缀图片"""
        if self.subfolder_list.selectedItems():
            selected_index = self.subfolder_list.row(item)
            folder_path = self.subfolders[selected_index]
            
            # 查找文件夹中的XXX_back后缀图片
            back_images = []
            try:
                for file in os.listdir(folder_path):
                    if file.endswith("_back.jpg") or file.endswith("_back.png") or file.endswith("_back.jpeg"):
                        try:
                            # 尝试提取XXX部分并转为数字
                            prefix = file.split("_back.")[0]
                            if prefix.isdigit():
                                back_images.append((int(prefix), file))
                        except:
                            # 如果转换失败，仍然添加该图片
                            back_images.append((0, file))
            except Exception as e:
                QMessageBox.warning(self, "错误", f"读取文件夹图片错误: {str(e)}")
                return
            
            # 如果找到图片，显示最大值的那张
            if back_images:
                back_images.sort(reverse=True)  # 按前缀值降序排序
                img_file = back_images[0][1]
                img_path = os.path.join(folder_path, img_file)
                self.current_image_path = img_path
                
                # 加载并显示图片
                self.load_image(img_path)
                self.image_info_label.setText(f"显示图片: {img_file} 来自: {folder_path}")
            else:
                self.image_label.clear()
                self.image_info_label.setText("未找到符合条件的图片")
                self.current_image_path = ""
    
    def load_image(self, image_path):
        """加载并自适应显示图片"""
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            # 获取滚动区域的大小
            scroll_size = self.scroll_area.size()
            # 缩放图片以适应滚动区域，保持原比例
            scaled_pixmap = pixmap.scaled(scroll_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.setText("无法加载图片")
    
    def create_new_folder(self):
        """创建新文件夹"""
        folder_name, ok = QInputDialog.getText(self, "新建文件夹", "请输入新文件夹名称:")
        if ok and folder_name:
            folder_path = QFileDialog.getExistingDirectory(self, "选择新文件夹位置", "")
            if folder_path:
                new_folder = os.path.join(folder_path, folder_name)
                try:
                    if not os.path.exists(new_folder):
                        os.makedirs(new_folder)
                        self.new_folder_path = new_folder
                        QMessageBox.information(self, "成功", f"已创建文件夹: {new_folder}")
                    else:
                        self.new_folder_path = new_folder
                        QMessageBox.information(self, "提示", f"文件夹已存在: {new_folder}")
                except Exception as e:
                    QMessageBox.warning(self, "错误", f"创建文件夹失败: {str(e)}")
    
    def move_folders(self):
        """将选中文件夹的所有内容移动到新文件夹"""
        if not self.new_folder_path:
            QMessageBox.warning(self, "错误", "请先创建新文件夹")
            return
        
        selected_items = self.subfolder_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "错误", "请先选择子文件夹")
            return
        
        moved_count = 0
        for item in selected_items:
            index = self.subfolder_list.row(item)
            source_folder_path = self.subfolders[index]
            folder_name = os.path.basename(source_folder_path)
            destination_folder_path = os.path.join(self.new_folder_path, folder_name)
            
            try:
                # 检查目标路径是否已存在
                if os.path.exists(destination_folder_path):
                    reply = QMessageBox.question(self, "确认覆盖", 
                                              f"目标文件夹 {destination_folder_path} 已存在，是否覆盖?",
                                              QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if reply == QMessageBox.Yes:
                        # 删除已存在的目标文件夹
                        shutil.rmtree(destination_folder_path)
                    else:
                        continue  # 跳过此文件夹
                
                # 复制整个文件夹到目标路径
                shutil.copytree(source_folder_path, destination_folder_path)
                moved_count += 1
            except Exception as e:
                QMessageBox.warning(self, "错误", f"移动文件夹 {folder_name} 失败: {str(e)}")
        
        if moved_count > 0:
            QMessageBox.information(self, "成功", f"已成功移动 {moved_count} 个文件夹到 {self.new_folder_path}")
        else:
            QMessageBox.warning(self, "提示", "未移动任何文件夹")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageViewer()
    window.show()
    sys.exit(app.exec_())