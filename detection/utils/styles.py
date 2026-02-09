def get_stylesheet():
    return """
    /* ===========================
       全局基础设置
       =========================== */
    QWidget {
        background-color: #f7f7f7;
        color: #333333;
        font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
        font-size: 16px;
    }

    /* ===========================
       按钮样式（扁平化）
       =========================== */
    QPushButton {
        background-color: #ffffff;
        color: #333333;
        border: 1px solid #cccccc;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: 600;
        min-width: 120px;
    }
    QPushButton:hover {
        background-color: #eeeeee;
    }
    QPushButton:pressed {
        background-color: #e0e0e0;
    }
    QPushButton:disabled {
        background-color: #f0f0f0;
        color: #aaaaaa;
        border: 1px solid #dddddd;
    }

    /* ===========================
       输入框(单行、多行、数值)样式
       =========================== */
    QLineEdit, QSpinBox, QTextEdit {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        border-radius: 4px;
        padding: 8px;
        selection-background-color: #cce8ff;
        selection-color: #333333;
        color: #333333;
        font-size: 16px;
    }
    QLineEdit:focus, QSpinBox:focus, QTextEdit:focus {
        border: 1px solid #3498db;
    }

    /* ===========================
       下拉框样式（扁平化）
       =========================== */
    QComboBox {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        border-radius: 4px;
        padding: 8px;
        min-width: 180px;
        color: #333333;
        font-size: 16px;
    }
    QComboBox:hover {
        border: 1px solid #3498db;
    }
    QComboBox::drop-down {
        border: none;
        width: 30px;
    }
    QComboBox::down-arrow {
        image: url(down_arrow.png);
        width: 16px;
        height: 16px;
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
       标签页样式（扁平化）
       =========================== */
    QTabWidget::pane {
        border: 1px solid #cccccc;
        border-radius: 4px;
        background-color: #ffffff;
    }
    QTabBar::tab {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        padding: 8px 16px;
        margin-right: 2px;
        font-size: 16px;
        color: #333333;
    }
    QTabBar::tab:selected {
        background-color: #ffffff; /* 页面名称区域背景色调整为白色 */
        border-bottom: 1px solid #f7f7f7;
        color: #333333;
    }

    /* ===========================
       进度条样式
       =========================== */
    QProgressBar {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        border-radius: 4px;
        text-align: center;
        height: 22px;
        font-size: 14px;
        color: #333333;
    }
    QProgressBar::chunk {
        background-color: #3498db;
        border-radius: 4px;
    }

    /* ===========================
       滚动条样式
       =========================== */
    QScrollBar:vertical {
        border: none;
        background-color: #f0f0f0;
        width: 10px;
        margin: 0;
    }
    QScrollBar::handle:vertical {
        background-color: #cccccc;
        border-radius: 4px;
        min-height: 30px;
    }
    QScrollBar::handle:vertical:hover {
        background-color: #bfbfbf;
    }

    /* ===========================
       单选按钮样式
       =========================== */
    QRadioButton {
        spacing: 8px;
        font-size: 16px;
    }
    QRadioButton::indicator {
        width: 18px;
        height: 18px;
    }

    /* ===========================
       复选框样式
       =========================== */
    QCheckBox {
        spacing: 8px;
        font-size: 16px;
    }
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
    }

    /* ===========================
       组框样式
       =========================== */
    QGroupBox {
        border: 1px solid #cccccc;
        border-radius: 4px;
        margin-top: 1em;
        padding-top: 1em;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 4px;
        font-size: 16px;
        color: #333333;
    }

    /* ===========================
       标签样式
       =========================== */
    QLabel {
        color: #333333;
        font-size: 16px;
        font-weight: 500;
    }

    /* ===========================
       特定控件自定义
       =========================== */
    #mainWindow {
        background-color: #f7f7f7;
    }
    .header-label {
        font-size: 20px;
        font-weight: bold;
        color: #333333;
    }
    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #333333;
    }
    """
