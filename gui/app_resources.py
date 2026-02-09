# -*- coding: utf-8 -*-
# app_resources.py
# 该文件包含应用程序的所有UI文本资源和参数解释，用于支持多语言切换。

RESOURCES = {
    'zh_cn': {
        'ui_texts': {
            # 窗口标题
            'window_title': "时序菌落分析工具 v{ui_version} (核心算法: HCP V{core_version})",
            # 模式选择
            'select_mode': "选择运行模式:",
            'mode_detection': "单文件夹检测",
            'mode_evaluation': "数据集评估",
            # 状态栏
            'status_initial': "请选择模式并加载数据",
            'status_loading_params': "参数文件 '{path}' 不存在，使用默认参数。",
            'status_params_loaded': "成功从 '{path}' 加载 {count} 个参数。",
            'status_params_saved': "参数已成功保存到 '{path}'。",
            'status_model_loading': "准备加载模型: {filename}...",
            'status_model_loading_wait': "正在加载模型，请稍候...",
            'status_model_load_success': "分类器模型 '{filename}' 加载成功。",
            'status_classifier_disabled': "分类器已禁用。显示HCP原始检测结果: {count}个。",
            'status_classifier_done': "分类完成。从 {initial_count} 个候选框中筛选出 {final_count} 个菌落。",
            # 按钮与标签
            'classifier_frame_title': "分类器过滤 (可选)",
            'classifier_enable_checkbox': "启用二分类模型过滤",
            'classifier_enable_checkbox_disabled': "启用过滤 (PyTorch未安装)",
            'classifier_load_button': "加载模型权重 (.pth)",
            'params_frame_title_detection': "算法参数 (检测模式)",
            'params_frame_title_evaluation': "算法参数 (评估模式)",
            'save_params_button': "保存当前参数",
            'load_params_button': "从文件加载参数",
            'load_folder_button': "加载图片文件夹",
            'start_processing_button': "开始处理",
            'save_results_button': "保存结果",
            # 新增：评估设置
            'eval_settings_group_title': "评估设置",
            'iou_threshold_label': "IoU 匹配阈值:",
            'perform_iou_sweep_checkbox': "执行IoU扫描评估 (0.05-0.95)",
            # 消息框
            'msgbox_exit_title': "退出程序",
            'msgbox_exit_content': "您确定要退出分析工具吗?",
            'error_title': "错误",
            'error_load_params': "加载参数文件 '{path}' 失败: {error}",
            'error_save_params': "保存参数文件失败: {error}",
            'error_load_model': "加载分类器模型失败: {error}",
            'error_torch_missing': "PyTorch未安装，无法加载模型。",
            'error_missing_model_args': "模型权重文件中缺少 'model_init_args'，无法自动重建模型。",
            # 进度更新
            'progress_stage_classifier': "分类器过滤",
            'progress_msg_classifier_prep': "准备过滤 {count} 个候选框...",
            'progress_msg_classifier_main': "已处理 {done}/{total} (当前保留: {kept})",
        },
        'param_tooltips': {
            # 预处理
            'num_bg_frames': "背景帧数: 用于构建背景模型的初始图像数量。建议值为5-15，取决于图像序列初期的稳定性。",
            'bf_diameter': "双边滤波-直径(d): 滤波时邻域的直径。奇数值，如5, 9。值越大，平滑效果越强，但会更慢。",
            'bf_sigmaColor': "双边滤波-颜色标准差(σ_c): 决定颜色空间中多少差异的像素会被混合。值越大，越远的颜色会被平滑。",
            'bf_sigmaSpace': "双边滤波-空间标准差(σ_s): 决定空间距离上多少差异的像素会被混合。值越大，越远的像素会相互影响。",
            'otsu_threshold_fallback_v2': "Otsu备用阈值: 当Otsu自适应阈值法失败时使用的固定阈值。用于从信号中分离前景。",
            'min_bbox_dimension_px': "最小边界框尺寸(像素): 任何边长小于此值的检测框将被忽略。",
            # Anchor 通道
            'anchor_channel': "主信号通道: 'negative'表示菌落比背景暗（常用），'positive'表示菌落比背景亮。决定了算法分析哪种信号。",
            'anchor_channel_threshold': "主信号阈值: 用于二值化主信号的阈值。设为0或负数时，算法将尝试使用Otsu自动计算。",
            # 泛晕处理
            'intelligent_halo_removal_enable': "智能泛晕去除: 是否启用。泛晕是菌落周围由光学效应产生的亮环。启用后可提高分割精度。",
            'halo_adjacency_dilation_px': "泛晕邻接膨胀(像素): 用于判断泛晕区域是否与菌落邻接的膨胀半径。建议2-5。",
            'halo_detection_overlap_threshold': "泛晕检测重叠阈值: 邻接的泛晕面积占总泛晕面积的比例。超过此阈值，则认为存在显著泛晕。",
            'halo_removal_erosion_px': "泛晕去除腐蚀(像素): 检测到显著泛晕后，对菌落掩码执行腐蚀操作的半径，以去除边缘的泛晕部分。",
            # 生物学验证
            'bio_validation_enable': "启用生物学验证: 是否基于生长趋势和形态学特征过滤候选目标。强烈建议启用。",
            'min_colony_area_px': "最小菌落面积(像素): 任何最终面积小于此值的对象都将被视为噪声并被移除。",
            'min_growth_slope_threshold': "最小生长斜率: Theil-Sen斜率估计出的像素平均强度增长率阈值。低于此值可能被认为是静态伪影。",
            'solidity_small_area_px': "小面积菌落定义(像素): 面积小于此值的对象被视为“小菌落”。",
            'solidity_small_threshold': "小菌落饱满度阈值: “小菌落”必须满足的最小饱满度(Solidity)。允许较低的值以适应早期不规则形态。",
            'solidity_medium_area_px': "中面积菌落定义(像素): 面积介于小、中之间的对象。",
            'solidity_medium_threshold': "中菌落饱满度阈值: “中等菌落”的最小饱满度要求。",
            'solidity_large_threshold': "大菌落饱满度阈值: “大菌落”的最小饱满度要求，通常要求形态非常规整。",
            'min_area_growth_rate': "最小面积增长率: 菌落区域内像素达到阈值的面积比例随时间变化的增长率。辅助判断是否在生长。",
            # 伪影分离
            'artifact_early_percentile': "伪影-早期出现百分位: 定义“早期出现”的时间点，基于所有信号首次出现时间的前百分之X。通常用于识别划痕等伪影。",
            'artifact_thin_distance_px': "伪影-细线结构距离(像素): 用于识别细线状伪影的膨胀半径。基于骨架化后的结构。",
            'artifact_circularity_threshold': "伪影-圆度阈值: 圆度低于此值的细长结构被怀疑是伪影。",
            'artifact_axis_ratio_threshold': "伪影-长宽比阈值: 长轴与短轴之比超过此值的细长结构被怀疑是伪影。",
            'overlap_ratio_threshold': "救援-重叠率阈值: 当一个检测对象与伪影掩码的重叠率低于此值时，直接保留该对象，不进行切割。",
            # 种子参数
            'seed_min_area_final': "最小种子面积(最终): 在种子提纯阶段，过滤掉面积过小的种子区域。通常设为1即可。"
        }
    },
    'en_us': {
        'ui_texts': {
            # Window Title
            'window_title': "Sequential Colony Analysis Tool v{ui_version} (Core: HCP V{core_version})",
            # Mode Selection
            'select_mode': "Select Mode:",
            'mode_detection': "Single Folder Detection",
            'mode_evaluation': "Dataset Evaluation",
            # Status Bar
            'status_initial': "Please select a mode and load data",
            'status_loading_params': "Params file '{path}' not found, using default values.",
            'status_params_loaded': "Successfully loaded {count} parameters from '{path}'.",
            'status_params_saved': "Parameters have been saved to '{path}'.",
            'status_model_loading': "Preparing to load model: {filename}...",
            'status_model_loading_wait': "Loading model, please wait...",
            'status_model_load_success': "Classifier model '{filename}' loaded successfully.",
            'status_classifier_disabled': "Classifier disabled. Showing {count} raw HCP detection results.",
            'status_classifier_done': "Classification finished. Filtered from {initial_count} candidates to {final_count} colonies.",
            # Buttons & Labels
            'classifier_frame_title': "Classifier Filter (Optional)",
            'classifier_enable_checkbox': "Enable Binary Classification Filter",
            'classifier_enable_checkbox_disabled': "Enable Filter (PyTorch not installed)",
            'classifier_load_button': "Load Model Weights (.pth)",
            'params_frame_title_detection': "Algorithm Parameters (Detection Mode)",
            'params_frame_title_evaluation': "Algorithm Parameters (Evaluation Mode)",
            'save_params_button': "Save Current Parameters",
            'load_params_button': "Load Parameters from File",
            'load_folder_button': "Load Image Folder",
            'start_processing_button': "Start Processing",
            'save_results_button': "Save Results",
            # New: Evaluation Settings
            'eval_settings_group_title': "Evaluation Settings",
            'iou_threshold_label': "IoU Match Threshold:",
            'perform_iou_sweep_checkbox': "Perform IoU Sweep Evaluation (0.05-0.95)",
            # Message Boxes
            'msgbox_exit_title': "Exit Application",
            'msgbox_exit_content': "Are you sure you want to exit the analysis tool?",
            'error_title': "Error",
            'error_load_params': "Failed to load parameter file '{path}': {error}",
            'error_save_params': "Failed to save parameter file: {error}",
            'error_load_model': "Failed to load classifier model: {error}",
            'error_torch_missing': "PyTorch is not installed, cannot load the model.",
            'error_missing_model_args': "Missing 'model_init_args' in model checkpoint, cannot rebuild model automatically.",
            # Progress Updates
            'progress_stage_classifier': "Classifier Filtering",
            'progress_msg_classifier_prep': "Preparing to filter {count} candidates...",
            'progress_msg_classifier_main': "Processed {done}/{total} (Kept: {kept})",
        },
        'param_tooltips': {
            # Pre-processing
            'num_bg_frames': "Number of BG Frames: The number of initial images used to build the background model. Recommended: 5-15, depending on initial stability.",
            'bf_diameter': "Bilateral Filter - Diameter (d): Diameter of each pixel neighborhood. Must be an odd integer, e.g., 5, 9. Larger values mean stronger smoothing but are slower.",
            'bf_sigmaColor': "Bilateral Filter - Sigma Color (σ_c): Determines how much different colors in the neighborhood will be mixed. A larger value means more distant colors will be smoothed.",
            'bf_sigmaSpace': "Bilateral Filter - Sigma Space (σ_s): Determines how far pixels influence each other. A larger value means more distant pixels will affect each other.",
            'otsu_threshold_fallback_v2': "Otsu Fallback Threshold: A fixed threshold used when the Otsu adaptive method fails. It separates foreground from the signal.",
            'min_bbox_dimension_px': "Min BBox Dimension (px): Bounding boxes with any side smaller than this value will be ignored.",
            # Anchor Channel
            'anchor_channel': "Anchor Channel: 'negative' means colonies are darker than the background (common), 'positive' means they are brighter. Determines which signal the algorithm analyzes.",
            'anchor_channel_threshold': "Anchor Channel Threshold: The threshold for binarizing the anchor signal. If set to 0 or negative, the algorithm will try to compute it automatically using Otsu.",
            # Halo Removal
            'intelligent_halo_removal_enable': "Intelligent Halo Removal: Toggles this feature. Halos are bright rings around colonies caused by optical effects. Enabling it improves segmentation accuracy.",
            'halo_adjacency_dilation_px': "Halo Adjacency Dilation (px): Dilation radius used to determine if a halo region is adjacent to a colony. Recommended: 2-5.",
            'halo_detection_overlap_threshold': "Halo Detection Overlap Threshold: The ratio of adjacent halo area to total halo area. If exceeded, significant halos are considered present.",
            'halo_removal_erosion_px': "Halo Removal Erosion (px): If significant halos are detected, this is the erosion radius applied to the colony mask to remove the halo edges.",
            # Biological Validation
            'bio_validation_enable': "Enable Biological Validation: Filters candidates based on growth trends and morphology. Strongly recommended to keep enabled.",
            'min_colony_area_px': "Min Colony Area (px): Any object with a final area smaller than this will be considered noise and removed.",
            'min_growth_slope_threshold': "Min Growth Slope: The threshold for the average pixel intensity growth rate, estimated by Theil-Sen robust regression. Objects below this may be static artifacts.",
            'solidity_small_area_px': "Small Area Colony Def (px): Objects with an area smaller than this are considered 'small colonies'.",
            'solidity_small_threshold': "Small Colony Solidity Threshold: The minimum solidity a 'small colony' must have. Lower values are allowed to accommodate irregular early-stage shapes.",
            'solidity_medium_area_px': "Medium Area Colony Def (px): Defines the area range for 'medium colonies'.",
            'solidity_medium_threshold': "Medium Colony Solidity Threshold: The minimum solidity requirement for 'medium colonies'.",
            'solidity_large_threshold': "Large Colony Solidity Threshold: The minimum solidity for 'large colonies', which are expected to be very regular.",
            'min_area_growth_rate': "Min Area Growth Rate: The growth rate of the proportion of pixels within a colony region that are above threshold over time. Helps confirm growth.",
            # Artifact Separation
            'artifact_early_percentile': "Artifact - Early Percentile: Defines the 'early appearance' time point based on the first X percentile of all signal appearance times. Used to identify scratches, etc.",
            'artifact_thin_distance_px': "Artifact - Thin Structure Dist (px): Dilation radius used to identify thin, line-like artifacts based on skeletonization.",
            'artifact_circularity_threshold': "Artifact - Circularity Threshold: Elongated structures with circularity below this value are suspected to be artifacts.",
            'artifact_axis_ratio_threshold': "Artifact - Axis Ratio Threshold: Elongated structures with a major-to-minor axis ratio above this value are suspected to be artifacts.",
            'overlap_ratio_threshold': "Rescue - Overlap Ratio Threshold: If a detected object's overlap with the artifact mask is below this ratio, it is kept directly without being split.",
            # Seed Parameters
            'seed_min_area_final': "Min Seed Area (Final): Filters out very small seed regions during the seed refinement stage. A value of 1 is usually sufficient."
        }
    }
}