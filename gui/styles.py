# -*- coding: utf-8 -*-
def get_stylesheet():
    return """
    /* ===========================
       全局基础设置 - 优化响应式布局
       =========================== */
    QWidget {
        background-color: #f7f7f7;
        color: #333333;
        font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
        font-size: 14px;
    }
    
    /* 统一布局容器基础样式 */
    QMainWindow, QWidget#centralWidget {
        background-color: #f7f7f7;
    }
    
    /* 统一边距和间距设置 */
    QVBoxLayout, QHBoxLayout {
        margin: 8px;
        spacing: 8px;
    }

    /* ===========================
       按钮样式（紧凑统一）
       =========================== */
    QPushButton {
        background-color: #ffffff;
        color: #333333;
        border: 1px solid #cccccc;
        border-radius: 6px;
        padding: 6px 12px;
        font-weight: 500;
        font-size: 13px;
        min-width: 80px;
        min-height: 28px;
        max-height: 36px;
    }
    QPushButton:hover {
        background-color: #e8f4f8;
        border-color: #0066cc;
    }
    QPushButton:pressed {
        background-color: #d0e8f0;
        border-color: #004080;
    }
    QPushButton:disabled {
        background-color: #f5f5f5;
        color: #999999;
        border: 1px solid #e0e0e0;
    }
    
    /* 主要操作按钮 */
    QPushButton#primaryButton {
        background-color: #0066cc;
        color: white;
        border: 1px solid #0066cc;
        font-weight: 600;
        min-width: 100px;
    }
    QPushButton#primaryButton:hover {
        background-color: #0052a3;
        border-color: #0052a3;
    }
    QPushButton#primaryButton:pressed {
        background-color: #004080;
        border-color: #004080;
    }

    /* ===========================
       输入框样式（紧凑一致）
       =========================== */
    QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit {
        background-color: #ffffff;
        border: 1px solid #d0d0d0;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 13px;
        color: #333333;
        min-height: 24px;
        max-height: 28px;
        selection-background-color: #cce8ff;
        selection-color: #333333;
    }
    QTextEdit {
        min-height: 60px;
        max-height: none;
        padding: 6px 8px;
    }
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {
        border: 2px solid #0066cc;
        padding: 3px 7px; /* 补偿边框增加的像素 */
    }
    QTextEdit:focus {
        padding: 5px 7px;
    }

    /* ===========================
       下拉框样式（紧凑统一）
       =========================== */
    QComboBox {
        background-color: #ffffff;
        border: 1px solid #d0d0d0;
        border-radius: 4px;
        padding: 4px 8px;
        min-width: 100px;
        min-height: 24px;
        max-height: 28px;
        color: #333333;
        font-size: 13px;
    }
    QComboBox:hover {
        border: 1px solid #0066cc;
    }
    QComboBox:focus {
        border: 2px solid #0066cc;
        padding: 3px 7px;
    }
    QComboBox::drop-down {
        border: none;
        width: 20px;
        background: transparent;
    }
    QComboBox::down-arrow {
        width: 12px;
        height: 12px;
        background: #666666;
        /* 使用CSS三角形替代图片 */
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 4px solid #666666;
        border-radius: 0px;
    }
    QComboBox QAbstractItemView {
        border: 1px solid #d0d0d0;
        selection-background-color: #e8f4f8;
        background-color: #ffffff;
        font-size: 13px;
    }

    /* ===========================
       列表和树状图样式
       =========================== */
    QListWidget, QTreeWidget {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        border-radius: 4px;
        font-size: 16px;
    }
    QListWidget::item, QTreeWidget::item {
        padding: 8px;
        border-bottom: 1px solid #efefef;
    }
    QListWidget::item:selected, QTreeWidget::item:selected {
        background-color: #3498db;
        color: #ffffff;
    }

    /* ===========================
       组框样式（紧凑统一）
       =========================== */
    QGroupBox {
        border: 1px solid #d0d0d0;
        border-radius: 6px;
        margin-top: 8px;
        padding-top: 8px;
        font-size: 13px;
        font-weight: 500;
        background-color: #fafafa;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
        font-size: 13px;
        font-weight: 600;
        color: #333333;
        background-color: #fafafa;
    }
    
    /* 紧凑的组框内边距 */
    QGroupBox > QWidget {
        margin: 4px;
    }
    
    /* Tab控件样式优化 */
    QTabWidget::pane {
        border: 1px solid #d0d0d0;
        border-radius: 4px;
        background-color: #ffffff;
        padding: 4px;
    }
    QTabBar::tab {
        background-color: #f5f5f5;
        border: 1px solid #d0d0d0;
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        padding: 6px 12px;
        margin-right: 1px;
        font-size: 13px;
        color: #666666;
        min-width: 60px;
    }
    QTabBar::tab:selected {
        background-color: #ffffff;
        border-bottom: 1px solid #ffffff;
        color: #333333;
        font-weight: 500;
    }
    QTabBar::tab:hover:!selected {
        background-color: #e8f4f8;
        color: #333333;
    }

    /* ===========================
       进度条和状态控件样式
       =========================== */
    QProgressBar {
        background-color: #f0f0f0;
        border: 1px solid #d0d0d0;
        border-radius: 4px;
        text-align: center;
        height: 20px;
        font-size: 12px;
        color: #333333;
    }
    QProgressBar::chunk {
        background-color: #0066cc;
        border-radius: 3px;
        margin: 1px;
    }
    
    /* 单选按钮和复选框样式 */
    QRadioButton, QCheckBox {
        spacing: 6px;
        font-size: 13px;
        color: #333333;
    }
    QRadioButton::indicator, QCheckBox::indicator {
        width: 16px;
        height: 16px;
    }
    QRadioButton::indicator::unchecked {
        border: 2px solid #d0d0d0;
        border-radius: 8px;
        background-color: #ffffff;
    }
    QRadioButton::indicator::checked {
        border: 2px solid #0066cc;
        border-radius: 8px;
        background-color: #ffffff;
        /* 中心圆点 */
        background-image: radial-gradient(circle, #0066cc 0px, #0066cc 4px, transparent 4px);
    }
    QCheckBox::indicator::unchecked {
        border: 2px solid #d0d0d0;
        border-radius: 2px;
        background-color: #ffffff;
    }
    QCheckBox::indicator::checked {
        border: 2px solid #0066cc;
        border-radius: 2px;
        background-color: #0066cc;
        /* Use Qt built-in resources for the check mark (avoid noisy "Could not create pixmap" warnings). */
        image: url(:/qt-project.org/styles/commonstyle/images/checkbox_checked.png);
    }

    /* ===========================
       滚动条和列表控件样式
       =========================== */
    QScrollBar:vertical {
        border: none;
        background-color: #f5f5f5;
        width: 8px;
        margin: 0;
        border-radius: 4px;
    }
    QScrollBar::handle:vertical {
        background-color: #d0d0d0;
        border-radius: 4px;
        min-height: 20px;
        margin: 1px;
    }
    QScrollBar::handle:vertical:hover {
        background-color: #b0b0b0;
    }
    QScrollBar::handle:vertical:pressed {
        background-color: #909090;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    
    /* 列表控件 */
    QListWidget, QTreeWidget {
        background-color: #ffffff;
        border: 1px solid #d0d0d0;
        border-radius: 4px;
        font-size: 13px;
        alternate-background-color: #f8f8f8;
    }
    QListWidget::item, QTreeWidget::item {
        padding: 4px 8px;
        border-bottom: 1px solid #f0f0f0;
        min-height: 20px;
    }
    QListWidget::item:selected, QTreeWidget::item:selected {
        background-color: #e8f4f8;
        color: #333333;
        border: 1px solid #0066cc;
    }
    QListWidget::item:hover, QTreeWidget::item:hover {
        background-color: #f0f8ff;
    }
    
    /* 标签样式 */
    QLabel {
        color: #333333;
        font-size: 13px;
        font-weight: 400;
        padding: 2px;
    }
    QLabel.header-label {
        font-size: 16px;
        font-weight: 600;
        color: #333333;
        padding: 4px 0px;
    }
    QLabel.section-title {
        font-size: 14px;
        font-weight: 500;
        color: #333333;
        padding: 2px 0px;
    }

    /* ===========================
       响应式布局和特定控件优化
       =========================== */
    #mainWindow {
        background-color: #f7f7f7;
    }
    
    /* 响应式容器 */
    QSplitter {
        background-color: #f7f7f7;
        border: none;
    }
    QSplitter::handle {
        background-color: #e0e0e0;
        border: 1px solid #d0d0d0;
        margin: 1px;
    }
    QSplitter::handle:horizontal {
        width: 6px;
        min-width: 6px;
        max-width: 8px;
    }
    QSplitter::handle:vertical {
        height: 6px;
        min-height: 6px;
        max-height: 8px;
    }
    QSplitter::handle:hover {
        background-color: #c0c0c0;
    }
    
    /* 紧凑的工具栏样式 */
    QFrame {
        border: none;
        background-color: transparent;
    }
    QFrame.panel {
        border: 1px solid #d0d0d0;
        border-radius: 4px;
        background-color: #ffffff;
        padding: 4px;
    }
    
    /* 状态栏优化 */
    QStatusBar {
        background-color: #f0f0f0;
        border-top: 1px solid #d0d0d0;
        font-size: 12px;
        color: #666666;
        padding: 2px 8px;
    }
    
    /* 菜单栏优化 */
    QMenuBar {
        background-color: #f7f7f7;
        border: none;
        font-size: 13px;
        padding: 2px;
    }
    QMenuBar::item {
        background-color: transparent;
        padding: 4px 8px;
        border-radius: 3px;
    }
    QMenuBar::item:selected {
        background-color: #e8f4f8;
        color: #333333;
    }
    
    /* 工具提示优化 */
    QToolTip {
        background-color: #333333;
        color: #ffffff;
        border: none;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 12px;
    }
    
    /* 自适应布局辅助类 */
    .compact-spacing {
        margin: 4px;
        padding: 4px;
    }
    .responsive-container {
        min-width: 200px;
        max-width: 100%;
    }
    """
