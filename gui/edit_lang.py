
# edit_lang.py
# -*- coding: utf-8 -*-

def get_editor_translation(lang):
    """
    返回指定语言的可视化编辑器翻译文本。

    Args:
        lang (str): 语言代码，如 'zh_CN' 或 'en'。

    Returns:
        dict: 包含UI元素翻译文本的字典。
    """
    # Normalize language codes: accept 'en_us'/'en-US' etc.
    try:
        lang_key = str(lang or 'en').strip()
    except Exception:
        lang_key = 'en'
    lang_key_low = lang_key.lower().replace('-', '_')
    if lang_key_low.startswith('zh'):
        lang_key = 'zh_CN'
    elif lang_key_low.startswith('en'):
        lang_key = 'en'
    else:
        lang_key = 'en'

    translations = {
        'zh_CN': {
            'window_title': '可视化注释编辑器',
            'open_folder_btn': '打开文件夹',
            'prev_sequence_btn': '上一序列',
            'next_sequence_btn': '下一序列',
            'add_bbox_btn': '添加边界框',
            'add_bbox_btn_exit_draw': '退出绘制',
            'delete_bbox_btn': '删除选定边界框',
            'delete_sequence_btn': '删除序列',
            'undo_btn': '撤销',
            'redo_btn': '重做',
            'save_btn': '保存更改',
            'add_category_btn': '添加类别',
            'delete_category_btn': '删除类别',
            'export_btn': '导出分类数据集',
            'help_btn': '说明',
            'toggle_enhanced_btn_off': '增强模式(关)',
            'toggle_enhanced_btn_on': '增强模式(开)',
            'set_interference_path_btn': '设置干扰保存路径',
            'set_interference_path_btn_dialog_title': '选择干扰框保存路径',
            'scale_bbox_btn': '批量缩放',
            'scale_bbox_btn_exit': '退出缩放',
            'prev_image_tooltip': '上一张图片',
            'next_image_tooltip': '下一张图片',
            'zoom_in_tooltip': '放大',
            'zoom_out_tooltip': '缩小',
            'label_label': '标签：',
            'apply_label_btn_selected': '应用标签(选中框)',
            'apply_label_btn_all': '应用标签(当前序列)',
            'language_combo_label': '语言：',
            'interference_label': '干扰框设置：',
            'enhanced_mode_label': '增强模式设置：',
            'compare_mode_label': '对比模式设置：',
            'compare_style_label': '对比样式：',
            'compare_style_split': '分屏(首帧 vs 末帧)',
            'compare_style_add_green': '叠加(绿色高亮)',
            'compare_style_add_rgb': '叠加(RGB)',
            'compare_style_blend': '混合(平均)',
            'compare_style_diff': '差分(差异图)',
            'toggle_compare_btn_off': '对比模式(关)',
            'toggle_compare_btn_on': '对比模式(开)',
            'compare_opacity_label': '叠加透明度(末帧叠到首帧):',
            'compare_first_frame_label': '首帧',
            'compare_last_frame_label': '末帧(标注)',
            'report_label': '数据集报告：',
            'graph_label': '统计图表：',
            'shortcut_label': '快捷键提示： [滚轮]缩放图像 | [右键]删除单个标注 | [Delete]删除选中 | [空格]切换绘图模式 | [左键框选]批量选中',
            'help_dialog_title': '说明',
            'help_dialog_text': (
                "<b>可视化注释编辑器使用说明</b><br><br>"
                "<b>基本操作：</b><br>"
                "1. 在非绘制模式下，按住左键并拖动鼠标，可批量选择标注框。<br>"
                "2. 选中一个或多个标注框后，点击'删除选定边界框'或按下Delete键即可删除。<br>"
                "3. 点击'删除序列'按钮，将删除当前序列的所有标注信息及对应图像文件。<br>"
                "4. 使用鼠标滚轮前后滚动，可放大或缩小图像。<br>"
                "5. 使用导航按钮可以左右上下平移图像，也可以使用+/-按钮放大缩小图像。<br><br>"
                "<b>图像浏览：</b><br>"
                "1. 使用上方的◀和▶按钮可以浏览当前序列的所有图像。<br>"
                "2. 中间的数字显示当前图像在序列中的位置。<br><br>"
                "<b>标注操作：</b><br>"
                "1. 在标签下拉框中选择类别。<br>"
                "2. 点击'添加边界框'或按空格键，在图像上按住左键拖拽绘制矩形标注框；再次点击或按空格键即可退出绘制模式。<br>"
                "3. 右键点击已有标注框可直接删除。<br><br>"
                "<b>数据操作与备份：</b><br>"
                "1. 点击'保存更改'可手动保存。每次保存时，程序会自动将旧的 `annotations.json` 备份为 `annotations.json.bak`。<br>"
                "2. 当有未保存的更改时，窗口标题会以 `[*]` 号提示。关闭程序时也会有保存提示。<br>"
                "3. 点击'导出分类数据集'选择输出目录，会裁剪所有标注框并生成分类数据集。<br><br>"
                "<b>类别管理：</b><br>"
                "1. '添加类别'可新增类别。<br>"
                "2. '删除类别'会移除该类别及相关标注。<br><br>"
                "<b>增强模式：</b><br>"
                "1. 若存在 images2 文件夹，点击'增强模式'按钮可在不同拍摄角度的图像间切换。<br>"
                "2. 标注数据仍共享同一份 annotations，无需重复编辑。<br><br>"
                "<b>干扰框保存：</b><br>"
                "1. 在右侧'干扰框设置'处，点击'设置干扰保存路径'可指定一个文件夹，用于存放所有被删除的标注框时序。<br>"
                "2. 无论是单个删除还是批量删除，都会将被删的标注框裁剪后存放于此。可用于训练中的干扰排除。<br><br>"
                "<b>批量缩放：</b><br>"
                "1. 点击'批量缩放'进入该模式。<br>"
                "2. 在图像上拖拽出一个矩形，选中该区域内的所有标注框。<br>"
                "3. 随后会出现一个黄色虚线框，拖动其边缘或角落可整体缩放所有选中的标注框。<br>"
                "4. 调整完毕后，再次点击'退出缩放'按钮即可应用更改。<br><br>"
                "<b>统计与报告：</b><br>"
                "1. 右侧实时显示统计报告和图表。<br>"
                "2. 会在 statistics 文件夹生成统计报告和图表。<br><br>"
                "<b>日志与帮助：</b><br>"
                "1. 日志文件位于 logs 文件夹。<br>"
                "2. 点击'说明'可再次查看帮助。<br>"
            ),
            'unsaved_changes_title': '未保存的更改',
            'unsaved_changes_text': '您有未保存的更改，是否在退出前保存？',
            'warning_no_folder': '请先打开一个文件夹。',
            'warning_no_categories_to_delete': '没有可删除的类别。',
            'warning_select_bbox_to_delete': '请选择要删除的边界框。',
            'warning_select_label_category': '请选择一个标签类别。',
            'warning_select_bbox_to_scale': '请先选择要缩放的边界框。',
            'warning_category_exists': '该类别已存在。',
            'warning_no_sequence_selected': '当前没有选择序列。',
            'warning_sequence_not_found': '无法找到当前序列。',
            'warning_no_image_sequence': '没有找到任何图像序列。',
            'warning_images_folder_not_found': '未找到 images 文件夹：{}',
            'warning_annotations_file_not_found': '未找到 annotations.json 文件：{}',
            'warning_load_annotations_failed': '无法加载 annotations.json：{}',
            'warning_sequence_folder_not_digit': "序列文件夹 '{}' 非纯数字，跳过",
            'warning_no_images_in_sequence': "序列 {} 中没有图像",
            'warning_image_path_not_found': "未找到图像路径 {} 对应的 image_id",
            'warning_no_bbox_in_selection': "当前选择中没有边界框。",
            'info_annotations_saved': '注释已成功保存。',
            'info_interference_path_set': '已设置干扰框保存路径：{}',
            'info_no_images2_folder': '当前没有检测到 images2 文件夹。',
            'info_export_success': '分类数据集已成功导出。',
            'info_restored_sequence_annotation': '已恢复序列注释，但图像文件无法恢复。',
            'info_interference_saved': '干扰框序列保存完成。',
            'info_interference_sequence_saved': '干扰框序列 {} 保存成功。',
            'info_prepare_export_classification_dataset': '正在准备导出分类数据集，请稍候...',
            'info_processing_sequence': '正在处理序列',
            'info_cropping_bbox_sequence': '正在裁剪序列',
            'info_category': '类别',
            'info_saving_annotations_json_file': '正在保存 annotations.json 文件...',
            'info_classification_dataset_exported': '分类数据集已成功导出。',
            'info_sequence_report_updated': '已更新 sequence_report.json',
            'info_enter_scale_mode': "进入批量缩放模式。请在图上拖拽出一个矩形区域来选择要缩放的标注框。",
            'error_save_annotations_failed': '保存注释时发生错误: {}',
            'error_generate_stats_report_failed': '生成统计报告时发生错误: {}',
            'error_save_stats_graph_failed': '保存统计图表时发生错误: {}',
            'error_generate_sequence_report_failed': '生成序列级统计报告时发生错误: {}',
            'error_save_visualized_image_failed': '保存可视化图像时发生错误: {}',
            'error_export_classification_failed': '导出分类数据集时发生错误: {}',
            'error_export_classification_failed_msg': '分类数据集导出失败，请查看日志信息。',
            'error_delete_sequence_images': '删除序列图像时发生错误: {}',
            'error_delete_sequence_images2': '删除序列图像时（images2）发生错误: {}',
            'error_crop_deleted_sequence': '删除标注框裁剪序列时错误: {}',
            'error_crop_deleted_sequence_images2': '删除标注框裁剪序列时（images2）错误: {}',
            'error_save_interference_sequence': '保存干扰框时序时出错 (interf_seq={}, image={}): {}',
            'error_save_interference_sequence_images2': '干扰框序列裁剪(images2)错误 (interf_seq={}, image={}): {}',
            'error_open_image_get_size': '无法打开图像 {} 获取尺寸: {}',
            'error_crop_bbox_sequence_image': '裁剪边界框序列图像时出错 (seq_id={}, bbox_seq={}, image={}): {}',
            'error_load_existing_annotations_json': '加载已存在annotations.json失败，可能导致ID冲突: {}',
            'error_update_annotation': '更新注释时发生错误',
            'error_detection_dataset_path_invalid': '指定的目标检测数据集路径不存在或不是文件夹: {}',
            'error_create_export_folder': '无法创建导出文件夹: {}, 错误信息: {}',
            'error_no_sequence_loaded': '未加载到任何序列数据，请检查目标检测数据集。',
            'error_interference_saved_failed': '干扰框序列保存失败，请查看日志信息。',
            'warning_interference_thread_running': '干扰框保存线程已经在运行中，请稍后重试。',
            'warning_export_thread_running': '导出任务正在进行中，请稍后。',
            'warning_folder_not_found': '指定的文件夹路径不存在：{}',
            'warning_sequence_id_not_found': '找不到序列ID {} 的信息。',
            'warning_no_images_in_current_sequence': '序列 {} 中没有图像。',
            'warning_category_id_not_found': '未找到 category_id {}',
            'status_label_init': '就绪',
            'status_label_sequence': '序列',
            'status_label_annotations': '标注数',
            'status_label_categories': '类别',
            'status_label_none': '无',
            'status_label_no_sequence': '无序列',
            'label_combo_label': '标签:',
            'language_combo_label': '语言:',
            'add_category_dialog_title': '添加类别',
            'add_category_dialog_text': '输入新类别名称：',
            'delete_category_dialog_title': '删除类别',
            'delete_category_dialog_text': '选择要删除的类别：',
            'export_classification_dialog_title': '导出分类数据集',
            'dataset_description': '培养皿时序标注数据集',
            'dataset_description_classification': '培养皿时序标注分类数据集（带 images2 增强模式）',
            'report_total_sequences': '总序列数',
            'report_total_images': '总图像数',
            'report_total_annotations': '总注释数',
            'report_category_counts': '类别计数',
            'report_avg_annotations_per_sequence': '平均每个序列的注释数',
            'report_current_sequence_id': '当前序列 ID',
            'report_current_sequence_image': '当前序列图像',
            'report_current_sequence_annotations_count': '当前序列注释数',
            'report_current_sequence_category_counts': '当前序列类别计数',
            'graph_category_xlabel': '类别',
            'graph_count_ylabel': '计数',
            'graph_title': '类别分布',
        },
        'en': {
            'window_title': 'Visual Annotation Editor',
            'open_folder_btn': 'Open Folder',
            'prev_sequence_btn': 'Previous Sequence',
            'next_sequence_btn': 'Next Sequence',
            'add_bbox_btn': 'Add BBox',
            'add_bbox_btn_exit_draw': 'Exit Draw',
            'delete_bbox_btn': 'Delete Selected',
            'delete_sequence_btn': 'Delete Sequence',
            'undo_btn': 'Undo',
            'redo_btn': 'Redo',
            'save_btn': 'Save Changes',
            'add_category_btn': 'Add Category',
            'delete_category_btn': 'Delete Category',
            'export_btn': 'Export Dataset',
            'help_btn': 'Help',
            'toggle_enhanced_btn_off': 'Enhanced Mode(Off)',
            'toggle_enhanced_btn_on': 'Enhanced Mode(On)',
            'set_interference_path_btn': 'Set Interference Path',
            'set_interference_path_btn_dialog_title': 'Select Interference Box Save Path',
            'scale_bbox_btn': 'Scale BBoxes',
            'scale_bbox_btn_exit': 'Exit Scale',
            'prev_image_tooltip': 'Previous Image',
            'next_image_tooltip': 'Next Image',
            'zoom_in_tooltip': 'Zoom In',
            'zoom_out_tooltip': 'Zoom Out',
            'label_label': 'Label:',
            'apply_label_btn_selected': 'Apply Label to Selected',
            'apply_label_btn_all': 'Apply Label to All',
            'language_combo_label': 'Language:',
            'interference_label': 'Interference Box Settings:',
            'enhanced_mode_label': 'Enhanced Mode Settings:',
            'compare_mode_label': 'Compare Mode Settings:',
            'compare_style_label': 'Compare Style:',
            'compare_style_split': 'Split View (First vs Last)',
            'compare_style_add_green': 'Add (Green Highlight)',
            'compare_style_add_rgb': 'Add (RGB)',
            'compare_style_blend': 'Blend',
            'compare_style_diff': 'Difference',
            'toggle_compare_btn_off': 'Compare Mode(Off)',
            'toggle_compare_btn_on': 'Compare Mode(On)',
            'compare_opacity_label': 'Overlay Opacity (Last on First):',
            'compare_first_frame_label': 'First Frame',
            'compare_last_frame_label': 'Last Frame (Annotate)',
            'report_label': 'Dataset Report:',
            'graph_label': 'Statistics Chart:',
            'shortcut_label': 'Shortcuts: [Wheel]Zoom | [Right Click]Delete BBox | [Delete]Delete Selected | [Space]Toggle Draw | [Drag]Multi-Select',
            'help_dialog_title': 'Help',
            'help_dialog_text': (
                "<b>Visual Annotation Editor - User Guide</b><br><br>"
                "<b>Basic Operations:</b><br>"
                "1. In non-drawing mode, left-click and drag to select multiple bounding boxes.<br>"
                "2. After selecting one or more bounding boxes, click 'Delete Selected' or press Delete key to delete them.<br>"
                "3. Click 'Delete Sequence' to remove all annotations and corresponding image files for the current sequence.<br>"
                "4. Use the mouse wheel to zoom in or out of the image.<br>"
                "5. Use navigation buttons to pan the image, or use +/- buttons to zoom.<br><br>"
                "<b>Image Navigation:</b><br>"
                "1. Use the ◀ and ▶ buttons at the top to browse all images in the current sequence.<br>"
                "2. The number in the middle shows the current image position in the sequence.<br><br>"
                "<b>Annotation Operations:</b><br>"
                "1. Select a category from the label dropdown.<br>"
                "2. Click 'Add BBox' or press Space key, then drag on the image to draw a bounding box; click again or press Space key to exit drawing mode.<br>"
                "3. Right-click on an existing bounding box to delete it directly.<br><br>"
                "<b>Data Operations & Backup:</b><br>"
                "1. Click 'Save Changes' to manually save. The old `annotations.json` will be automatically backed up as `annotations.json.bak` each time you save.<br>"
                "2. The window title will show `[*]` for unsaved changes. You will be prompted to save when closing the application.<br>"
                "3. Click 'Export Dataset', choose an output directory to crop all bounding boxes and generate a classification dataset.<br><br>"
                "<b>Category Management:</b><br>"
                "1. 'Add Category' to add new categories.<br>"
                "2. 'Delete Category' will remove the category and related annotations.<br><br>"
                "<b>Enhanced Mode:</b><br>"
                "1. If an images2 folder exists, click 'Enhanced Mode' button to switch between images from different viewpoints.<br>"
                "2. Annotation data is shared, no need to re-edit.<br><br>"
                "<b>Interference Box Saving:</b><br>"
                "1. Click 'Set Interference Path' to specify a folder for saving all deleted bounding box sequences.<br>"
                "2. These can be used for interference exclusion in training.<br><br>"
                "<b>Batch Scaling:</b><br>"
                "1. Click 'Scale BBoxes' to enter the mode.<br>"
                "2. Drag a rectangle on the image to select all bounding boxes within it.<br>"
                "3. A yellow dashed box will appear. Drag its edges or corners to scale all selected boxes proportionally.<br>"
                "4. Click 'Exit Scale' to apply the changes.<br><br>"
                "<b>Statistics and Reports:</b><br>"
                "1. Real-time statistical reports and charts are displayed on the right.<br>"
                "2. Reports and charts will be saved in the 'statistics' folder.<br><br>"
                "<b>Logs and Help:</b><br>"
                "1. Log files are located in the 'logs' folder.<br>"
                "2. Click 'Help' to view this guide again.<br>"
            ),
            'unsaved_changes_title': 'Unsaved Changes',
            'unsaved_changes_text': 'You have unsaved changes. Would you like to save before exiting?',
            'warning_no_folder': 'Please open a folder first.',
            'warning_no_categories_to_delete': 'No categories to delete.',
            'warning_select_bbox_to_delete': 'Please select bounding boxes to delete.',
            'warning_select_label_category': 'Please select a label category.',
            'warning_select_bbox_to_scale': 'Please select bounding boxes to scale first.',
            'warning_category_exists': 'Category already exists.',
            'warning_no_sequence_selected': 'No sequence selected.',
            'warning_sequence_not_found': 'Current sequence not found.',
            'warning_no_image_sequence': 'No image sequences found.',
            'warning_images_folder_not_found': 'Images folder not found: {}',
            'warning_annotations_file_not_found': 'annotations.json file not found: {}',
            'warning_load_annotations_failed': 'Failed to load annotations.json: {}',
            'warning_sequence_folder_not_digit': "Sequence folder '{}' is not purely numeric, skipped",
            'warning_no_images_in_sequence': "No images in sequence {}",
            'warning_image_path_not_found': "Image path {}'s corresponding image_id not found",
            'warning_no_bbox_in_selection': "No bounding box in this selection.",
            'info_annotations_saved': 'Annotations saved successfully.',
            'info_interference_path_set': 'Interference box save path set to: {}',
            'info_no_images2_folder': 'images2 folder not detected.',
            'info_export_success': 'Classification dataset exported successfully.',
            'info_restored_sequence_annotation': 'Sequence annotation restored, but image files cannot be recovered.',
            'info_interference_saved': 'Interference box sequence saved successfully.',
            'info_interference_sequence_saved': 'Interference box sequence {} saved successfully.',
            'info_prepare_export_classification_dataset': 'Preparing to export classification dataset, please wait...',
            'info_processing_sequence': 'Processing sequence',
            'info_cropping_bbox_sequence': 'Cropping sequence',
            'info_category': 'Category',
            'info_saving_annotations_json_file': 'Saving annotations.json file...',
            'info_classification_dataset_exported': 'Classification dataset exported successfully.',
            'info_sequence_report_updated': 'sequence_report.json updated',
            'info_enter_scale_mode': "Entered batch scale mode. Drag a rectangle on the image to select bboxes for scaling.",
            'error_save_annotations_failed': 'Error saving annotations: {}',
            'error_generate_stats_report_failed': 'Error generating statistics report: {}',
            'error_save_stats_graph_failed': 'Error saving statistics graph: {}',
            'error_generate_sequence_report_failed': 'Error generating sequence-level statistics report: {}',
            'error_save_visualized_image_failed': 'Error saving visualized image: {}',
            'error_export_classification_failed': 'Error exporting classification dataset: {}',
            'error_export_classification_failed_msg': 'Classification dataset export failed, please check log information.',
            'error_delete_sequence_images': 'Error deleting sequence images: {}',
            'error_delete_sequence_images2': 'Error deleting sequence images (images2): {}',
            'error_crop_deleted_sequence': 'Error cropping sequence when deleting bbox: {}',
            'error_crop_deleted_sequence_images2': 'Error cropping sequence (images2) when deleting bbox: {}',
            'error_save_interference_sequence': 'Error saving interference box sequence (interf_seq={}, image={}): {}',
            'error_save_interference_sequence_images2': 'Error cropping interference box sequence (images2) (interf_seq={}, image={}): {}',
            'error_open_image_get_size': 'Failed to open image {} to get size: {}',
            'error_crop_bbox_sequence_image': 'Error cropping bbox sequence image (seq_id={}, bbox_seq={}, image={}): {}',
            'error_load_existing_annotations_json': 'Failed to load existing annotations.json, potential ID conflicts: {}',
            'error_update_annotation': 'Error updating annotation',
            'error_detection_dataset_path_invalid': 'Specified detection dataset path is invalid or not a folder: {}',
            'error_create_export_folder': 'Failed to create export folder: {}, Error message: {}',
            'error_no_sequence_loaded': 'No sequence data loaded, please check the detection dataset.',
            'error_interference_saved_failed': 'Interference box sequence saving failed, please check log information.',
            'warning_interference_thread_running': 'Interference box saving thread is already running, please try again later.',
            'warning_export_thread_running': 'Export task is running, please wait.',
            'warning_folder_not_found': 'Specified folder path does not exist: {}',
            'warning_sequence_id_not_found': 'Sequence ID {} information not found.',
            'warning_no_images_in_current_sequence': 'No images in sequence {}.',
            'warning_category_id_not_found': 'Category_id {} not found',
            'status_label_init': 'Ready',
            'status_label_sequence': 'Sequence',
            'status_label_annotations': 'Annotations',
            'status_label_categories': 'Categories',
            'status_label_none': 'None',
            'status_label_no_sequence': 'No Sequence',
            'label_combo_label': 'Label:',
            'language_combo_label': 'Language:',
            'add_category_dialog_title': 'Add Category',
            'add_category_dialog_text': 'Enter new category name:',
            'delete_category_dialog_title': 'Delete Category',
            'delete_category_dialog_text': 'Select category to delete:',
            'export_classification_dialog_title': 'Export Classification Dataset',
            'dataset_description': 'Culture dish sequence annotation dataset',
            'dataset_description_classification': 'Culture dish sequence annotation classification dataset (with images2 enhanced mode)',
            'report_total_sequences': 'Total Sequences',
            'report_total_images': 'Total Images',
            'report_total_annotations': 'Total Annotations',
            'report_category_counts': 'Category Counts',
            'report_avg_annotations_per_sequence': 'Avg Annotations per Sequence',
            'report_current_sequence_id': 'Current Sequence ID',
            'report_current_sequence_image': 'Current Sequence Image',
            'report_current_sequence_annotations_count': 'Current Sequence Annotation Count',
            'report_current_sequence_category_counts': 'Current Sequence Category Counts',
            'graph_category_xlabel': 'Category',
            'graph_count_ylabel': 'Count',
            'graph_title': 'Category Distribution',
        }
    }
    return translations.get(lang_key, translations['en'])  # 默认返回英文


def retranslate_editor_ui(editor, translation):
    """
    【已重构】使用提供的翻译文本重新翻译可视化编辑器的UI元素。
    移除了重复和无效的代码，确保所有组件都被正确、高效地翻译。

    Args:
        editor (AnnotationEditor): AnnotationEditor 实例。
        translation (dict): 包含翻译文本的字典。
    """
    # 窗口标题根据是否有未保存的更改来设置
    base_title = translation.get('window_title', 'Visual Annotation Editor')
    editor.setWindowTitle(f"{base_title}[*]" if editor.has_unsaved_changes else base_title)

    # 对话框标题/文本
    editor.help_dialog_title = translation.get('help_dialog_title', 'Help')
    editor.help_dialog_text = translation.get('help_dialog_text', '')
    if hasattr(editor, 'add_category_dialog_title'):
        editor.add_category_dialog_title = translation.get('add_category_dialog_title', 'Add Category')
    if hasattr(editor, 'add_category_dialog_text'):
        editor.add_category_dialog_text = translation.get('add_category_dialog_text', 'Enter new category name:')
    if hasattr(editor, 'delete_category_dialog_title'):
        editor.delete_category_dialog_title = translation.get('delete_category_dialog_title', 'Delete Category')
    if hasattr(editor, 'delete_category_dialog_text'):
        editor.delete_category_dialog_text = translation.get('delete_category_dialog_text', 'Select category to delete:')
    if hasattr(editor, 'export_classification_dialog_title'):
        editor.export_classification_dialog_title = translation.get('export_classification_dialog_title', 'Export Classification Dataset')
    if hasattr(editor, 'set_interference_path_btn_dialog_title'):
        editor.set_interference_path_btn_dialog_title = translation.get('set_interference_path_btn_dialog_title', 'Select Interference Box Save Path')

    # 按钮文本
    editor.open_folder_btn.setText(translation['open_folder_btn'])
    editor.prev_btn.setText(translation['prev_sequence_btn'])
    editor.next_btn.setText(translation['next_sequence_btn'])
    editor.delete_bbox_btn.setText(translation['delete_bbox_btn'])
    editor.delete_sequence_btn.setText(translation['delete_sequence_btn'])
    editor.undo_btn.setText(translation['undo_btn'])
    editor.redo_btn.setText(translation['redo_btn'])
    editor.save_btn.setText(translation['save_btn'])
    editor.add_category_btn.setText(translation['add_category_btn'])
    editor.delete_category_btn.setText(translation['delete_category_btn'])
    editor.export_btn.setText(translation['export_btn'])
    editor.help_btn.setText(translation['help_btn'])
    editor.set_interference_path_btn.setText(translation['set_interference_path_btn'])
    if hasattr(editor, 'apply_label_btn'):
        editor.apply_label_btn.setText(translation.get('apply_label_btn_all', 'Apply Label to All'))

    # 动态文本按钮
    if editor.use_enhanced_view:
        editor.toggle_enhanced_btn.setText(translation['toggle_enhanced_btn_on'])
    else:
        editor.toggle_enhanced_btn.setText(translation['toggle_enhanced_btn_off'])
    if hasattr(editor, 'toggle_compare_btn'):
        if getattr(editor, 'compare_mode', False):
            editor.toggle_compare_btn.setText(translation.get('toggle_compare_btn_on', 'Compare Mode(On)'))
        else:
            editor.toggle_compare_btn.setText(translation.get('toggle_compare_btn_off', 'Compare Mode(Off)'))

    # Compare style items
    if hasattr(editor, 'compare_style_combo'):
        current_data = editor.compare_style_combo.currentData()
        editor.compare_style_combo.blockSignals(True)
        editor.compare_style_combo.clear()
        editor.compare_style_combo.addItem(translation.get('compare_style_split', 'Split View (First vs Last)'), "split")
        editor.compare_style_combo.addItem(translation.get('compare_style_add_green', 'Add (Green Highlight)'), "add_green")
        editor.compare_style_combo.addItem(translation.get('compare_style_add_rgb', 'Add (RGB)'), "add_rgb")
        editor.compare_style_combo.addItem(translation.get('compare_style_blend', 'Blend'), "blend")
        editor.compare_style_combo.addItem(translation.get('compare_style_diff', 'Difference'), "diff")
        idx = editor.compare_style_combo.findData(current_data)
        if idx >= 0:
            editor.compare_style_combo.setCurrentIndex(idx)
        editor.compare_style_combo.blockSignals(False)

    if editor.is_batch_scaling:
        editor.scale_bbox_btn.setText(translation.get('scale_bbox_btn_exit', 'Exit Scale'))
    else:
        editor.scale_bbox_btn.setText(translation.get('scale_bbox_btn', 'Scale BBoxes'))
    
    # 修复：确保 editor.graphics_view 存在且是我们自定义的类
    if hasattr(editor, 'graphics_view') and hasattr(editor.graphics_view, 'drawing'):
        if editor.graphics_view.drawing:
            editor.add_bbox_btn.setText(translation.get('add_bbox_btn_exit_draw', 'Exit Draw'))
        else:
            editor.add_bbox_btn.setText(translation.get('add_bbox_btn', 'Add BBox'))

    # 标签文本
    editor.label_label.setText(translation['label_label'])
    editor.interference_label.setText(translation['interference_label'])
    editor.enhanced_mode_label.setText(translation['enhanced_mode_label'])
    if hasattr(editor, 'compare_mode_label'):
        editor.compare_mode_label.setText(translation.get('compare_mode_label', 'Compare Mode Settings:'))
    if hasattr(editor, 'compare_style_label'):
        editor.compare_style_label.setText(translation.get('compare_style_label', 'Compare Style:'))
    if hasattr(editor, 'compare_opacity_label'):
        editor.compare_opacity_label.setText(translation.get('compare_opacity_label', 'Overlay Opacity (Last on First):'))
    if hasattr(editor, 'compare_title_label'):
        editor.compare_title_label.setText(translation.get('compare_first_frame_label', 'First Frame'))
    if hasattr(editor, 'main_title_label'):
        editor.main_title_label.setText(translation.get('compare_last_frame_label', 'Last Frame (Annotate)'))
    editor.report_label.setText(translation['report_label'])
    editor.graph_label.setText(translation['graph_label'])
    editor.shortcut_label.setText(translation['shortcut_label'])
    if hasattr(editor, 'language_combo_label'):
        editor.language_combo_label.setText(translation.get('language_combo_label', 'Language:'))
    if hasattr(editor, 'label_combo_label'):
        editor.label_combo_label.setText(translation.get('label_combo_label', 'Label:'))

    # 按钮提示 (Tooltips)
    if hasattr(editor, 'prev_image_btn'):
        editor.prev_image_btn.setToolTip(translation.get('prev_image_tooltip', 'Previous Image'))
    if hasattr(editor, 'next_image_btn'):
        editor.next_image_btn.setToolTip(translation.get('next_image_tooltip', 'Next Image'))
    if hasattr(editor, 'zoom_in_btn'):
        editor.zoom_in_btn.setToolTip(translation.get('zoom_in_tooltip', 'Zoom In'))
    if hasattr(editor, 'zoom_out_btn'):
        editor.zoom_out_btn.setToolTip(translation.get('zoom_out_tooltip', 'Zoom Out'))
    
    # 刷新动态内容的UI
    editor.update_status_label()
    editor.update_statistical_graph()
    editor.update_dataset_report()
