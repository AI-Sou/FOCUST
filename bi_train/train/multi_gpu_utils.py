# train/multi_gpu_utils.py
# 全面优化的多GPU支持模块 - 激进版，最大化GPU利用率

import os
import sys
import json
import logging
import pynvml
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
try:
    from .config_utils import get_message
except ImportError:  # pragma: no cover
    from train.config_utils import get_message

def setup_multi_gpu(model, config, logger=None, language='en'):
    """
    【已修复和简化】的标准版多GPU设置函数。
    - 不再需要复杂的递归来“修补”模型，因为模型定义本身已被修复。
    - 采用 PyTorch 推荐的标准 nn.DataParallel 包装流程。
    - 自动从 CUDA_VISIBLE_DEVICES 环境变量或所有可用GPU中获取设备列表。

    Args:
        model: 待分发的模型
        config: 配置字典
        logger: 日志记录器 (可选)
        language: 语言设置
        
    Returns:
        (model, is_multi_gpu): 处理后的模型和是否成功启用多GPU
    """
    # 检查是否启用多GPU以及系统是否有多个GPU
    if not config.get('use_multi_gpu', False):
        return model, False
        
    gpu_count = torch.cuda.device_count()
    if gpu_count <= 1:
        if logger:
            logger.info(get_message(language, "only_one_gpu_available"))
        return model, False
    
    # 确定要使用的GPU设备ID列表
    visible_devices_str = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible_devices_str:
        # 如果设置了环境变量，则使用它
        device_ids = [int(i) for i in visible_devices_str.split(',') if i.strip()]
        if logger:
            logger.info(f"从环境变量 CUDA_VISIBLE_DEVICES='{visible_devices_str}' 中解析出GPU IDs: {device_ids}")
    else:
        # 否则，使用所有可用的GPU
        device_ids = list(range(gpu_count))
        if logger:
            logger.info(f"未设置 CUDA_VISIBLE_DEVICES，将使用所有 {gpu_count} 个可用GPU: {device_ids}")
            
    if len(device_ids) <= 1:
        if logger:
            logger.info("过滤后只有一个或没有GPU可用，不启用DataParallel。")
        return model, False

    try:
        # 1. 将模型移动到主设备（列表中的第一个GPU）
        #    因为模型内部已修复，.to()会正确移动所有参数和缓冲区。
        main_device = f'cuda:{device_ids[0]}'
        model.to(main_device)
        if logger:
            logger.info(f"模型已移动到主设备: {main_device}")
            
        # 2. 使用 nn.DataParallel 包装模型
        #    DataParallel会自动将数据分发到 device_ids 中列出的所有GPU上，
        #    并在每个GPU上创建模型的副本。
        model = nn.DataParallel(model, device_ids=device_ids)
        
        if logger:
            logger.info(get_message(language, "multi_gpu_config_success").format(device_ids))
            
        return model, True
        
    except Exception as e:
        if logger:
            logger.error(get_message(language, "multi_gpu_config_fail").format(str(e)))
            import traceback
            logger.error(traceback.format_exc())
        # 如果失败，将模型移回CPU以防万一，并返回
        return model.to('cpu'), False


def is_multi_gpu_model(model):
    """
    判断模型是否在多个GPU上
    
    Args:
        model: 待检查的模型
        
    Returns:
        bool: 是否为多GPU模型
    """
    return isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel)

def get_model_without_ddp(model):
    """
    获取不带DataParallel或DistributedDataParallel包装的原始模型
    
    Args:
        model: 可能被包装的模型
        
    Returns:
        原始未包装模型
    """
    if is_multi_gpu_model(model):
        return model.module
    return model

def ensure_tensor_contiguous(tensor):
    """
    确保张量是连续的，以提高训练效率
    
    Args:
        tensor: 输入张量
        
    Returns:
        连续化后的张量
    """
    if not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor

def select_all_gpus(min_free_memory_mb=1024, max_gpu_memory_mb=25000, logger=None, language='en'):
    """
    【已修复】选择所有满足要求的GPU，并设置环境变量。
    新增逻辑：仅选择剩余显存大于总显存一半的GPU。
    
    Args:
        min_free_memory_mb: 最小可用显存要求(MB)
        max_gpu_memory_mb: 最大显存限制(MB)
        logger: 日志记录器
        language: 语言设置
        
    Returns:
        (selected_gpu_ids, available_memories): 选中的GPU ID列表和每个GPU的可用内存
    """
    selected_gpu_ids = []
    available_memories = {}
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        logger.info(get_message(language, "system_gpu_count").format(device_count))
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_free_mb = mem_info.free // (1024 * 1024)
            mem_total_mb = mem_info.total // (1024 * 1024)
            
            # 获取设备名称和使用率
            device_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu
            
            if logger:
                logger.info(get_message(language, "gpu_memory_info").format(i, device_name, mem_total_mb, mem_free_mb, gpu_util))
            
            # --- Bug修复：增加显存必须过半的判断条件 ---
            is_memory_sufficient = mem_free_mb >= min_free_memory_mb
            is_memory_more_than_half = mem_free_mb > (mem_total_mb / 2)
            
            if is_memory_sufficient and is_memory_more_than_half:
                selected_gpu_ids.append(i)
                available_memories[i] = min(mem_free_mb, max_gpu_memory_mb)
                if logger:
                    logger.info(f"GPU {i} 符合要求，将被选中。(可用显存: {mem_free_mb}MB > 总显存一半: {mem_total_mb/2:.0f}MB)")
            else:
                # 记录不满足条件的原因
                reasons = []
                if not is_memory_more_than_half:
                    reasons.append(f"可用显存 {mem_free_mb}MB 未超过总显存的一半({mem_total_mb/2:.0f}MB)")
                if not is_memory_sufficient:
                    reasons.append(f"可用显存 {mem_free_mb}MB 小于最低要求 {min_free_memory_mb}MB")
                
                if logger:
                    logger.info(f"GPU {i} 未被选中，原因: " + " & ".join(reasons))
                    
        pynvml.nvmlShutdown()
    except Exception as e:
        if logger:
            logger.error(get_message(language, "nvml_init_fail").format(e))
            logger.error(get_message(language, "gpu_info_error").format(str(e)))
        else:
            print(get_message(language, "nvml_init_fail").format(e))
        return [], {}

    if not selected_gpu_ids:
        if logger:
            logger.warning(get_message(language, "no_gpu_available").format(min_free_memory_mb))
        else:
            print(get_message(language, "no_gpu_available").format(min_free_memory_mb))
        return [], {}
    else:
        # 设置环境变量，只使用选定的GPU
        gpu_str = ','.join(map(str, selected_gpu_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        if logger:
            logger.info(get_message(language, "selected_gpus").format(gpu_str))
            logger.info(get_message(language, "cuda_visible_devices_set").format(gpu_str))
            for gpu_id in selected_gpu_ids:
                logger.info(get_message(language, "gpu_memory_available").format(gpu_id, available_memories[gpu_id]))
        return selected_gpu_ids, available_memories

def estimate_batch_size_multi_gpu(config, model, device, logger, available_memories, train_dataset, language='en', task_type='classification', aggressive_mode=True):
    """
    增强版多GPU批次大小估算 - 激进版本，最大化GPU利用率
    关键改进：
    1. 更激进的内存阈值，最大程度使用可用GPU显存
    2. 通信开销估计更贴近实际值，避免过度保守
    3. 搜索策略更积极，优先使用更大批次大小
    4. 支持高级内存优化选项，允许用户选择更激进的策略
    5. 出错时自动重试，更加鲁棒
    
    Args:
        config: 配置字典
        model: 模型实例
        device: 计算设备
        logger: 日志记录器
        available_memories: 每个GPU的可用显存
        train_dataset: 训练数据集
        language: 语言设置
        task_type: 任务类型
        aggressive_mode: 是否启用激进模式，最大化利用显存
        
    Returns:
        更新后的配置字典
    """
    if device.type != 'cuda':
        logger.info(get_message(language, "current_device_not_gpu"))
        return config

    # 如果只有一个GPU，回退到单GPU批次大小估算
    if len(available_memories) <= 1:
        try:
            from .train_utils import adjust_batch_size_based_on_memory
        except ImportError:  # pragma: no cover
            from train.train_utils import adjust_batch_size_based_on_memory
        return adjust_batch_size_based_on_memory(config, model, device, logger, list(available_memories.values())[0], train_dataset, language, task_type)

    # 计算每个GPU的总显存和最小可用显存
    total_memory = sum(available_memories.values())
    min_gpu_memory = min(available_memories.values())
    gpu_count = len(available_memories)
    
    # 更激进的内存阈值，最大程度利用GPU显存
    # 激进模式使用更高的内存阈值，同时考虑GPU数量的影响
    if aggressive_mode:
        # 激进模式: 高内存阈值，利用更多可用显存
        memory_threshold = 1.3 - (0.01 * (gpu_count - 1))  # 从85%开始，每增加一个GPU降低1%
        memory_threshold = max(0.90, memory_threshold)      # 不低于75%
    else:
        # 标准模式: 较保守的阈值，保留更多安全余量
        memory_threshold = 0.75 - (0.02 * (gpu_count - 1))  # 从75%开始，每增加一个GPU降低2%
        memory_threshold = max(0.65, memory_threshold)      # 不低于65%
    
    # 初始批次大小和最大批次大小 - 激进版本
    initial_bs = config.get('batch_size', 4)
    
    # 更高的最大批次大小上限，最大化利用多GPU
    max_bs = initial_bs * gpu_count * 8  # 较高上限，最大化利用
    
    # 初始步长 - 激进版本使用更大的初始步长
    step = max(initial_bs, 4)  # 更大的初始步长，加速搜索
    
    logger.info(f"【激进模式】启动多GPU批次大小优化: 初始={initial_bs}, 最大={max_bs}, GPU数量={gpu_count}")
    logger.info(f"【激进模式】显存情况: 总可用={total_memory}MB, 最小GPU可用={min_gpu_memory}MB")
    logger.info(f"【激进模式】使用内存阈值: {memory_threshold*100:.1f}% (目标显存占用率)")

    current_bs = initial_bs
    best_bs = initial_bs
    
    # 记录已经尝试过的大小，避免重复计算
    tried_sizes = set()
    
    # 记录最后一次成功的估算结果
    last_successful_est = 0
    
    # 预测启动失败的情况下，允许多次重试
    retries_left = 3
    
    while current_bs <= max_bs:
        if current_bs in tried_sizes:
            current_bs += step
            continue
            
        tried_sizes.add(current_bs)
        try:
            logger.info(f"【激进模式】尝试总批次大小={current_bs}")
            
            # 每个GPU的批次大小，确保均匀分配
            per_gpu_bs = max(1, current_bs // gpu_count)
            
            # 构建输入形状
            if task_type == 'classification':
                input_shape = (per_gpu_bs, config.get('max_seq_length', 40), 3, 224, 224)
            else:
                input_shape = (per_gpu_bs, config.get('max_seq_length', 40), 3, 4000, 4000)
                
            # 估算显存使用
            if config['data_mode'] == 'enhanced':
                est = estimate_model_memory_per_gpu(model, device, input_size=(input_shape, input_shape), aggressive_mode=aggressive_mode)
            else:
                est = estimate_model_memory_per_gpu(model, device, input_size=input_shape, aggressive_mode=aggressive_mode)
                
            # 更高效的通信开销计算 - 激进版本
            # 使用较低的通信开销估计，避免过度保守
            # 针对激进模式进一步降低估计的通信开销
            if aggressive_mode:
                # 激进模式通信因子，更低的估计值
                communication_overhead = 1.0 + (0.08 * math.log2(gpu_count))  # 更小的对数系数
            else:
                # 默认通信因子
                communication_overhead = 1.0 + (0.12 * math.log2(gpu_count))
            
            # 批次大小影响因子 - 激进版本更低
            batch_factor = 1.0 + (0.01 * (per_gpu_bs - 1))  # 每增加1的批次大小增加1%开销(更低)
            batch_factor = min(1.15, batch_factor)  # 最多增加15%(更低)
            
            est_with_overhead = est * communication_overhead * batch_factor
            
            logger.info(f"【激进模式】基础显存估计: {est:.2f}MB")
            logger.info(f"【激进模式】通信开销因子: {communication_overhead:.2f}x")
            logger.info(f"【激进模式】批次大小因子: {batch_factor:.2f}x") 
            logger.info(f"【激进模式】估计总显存需求: {est_with_overhead:.2f}MB (每GPU), 可用显存={min_gpu_memory}MB")
            logger.info(f"【激进模式】显存利用率: {est_with_overhead/min_gpu_memory*100:.1f}% (阈值: {memory_threshold*100:.1f}%)")

            # 使用更激进的阈值判断是否超过显存限制
            if est_with_overhead > min_gpu_memory * memory_threshold:
                logger.warning(f"【激进模式】批次大小={current_bs}超过显存阈值，停止扩大")
                
                # 如果非常接近阈值，尝试以更小步长继续
                threshold_proximity = est_with_overhead / (min_gpu_memory * memory_threshold)
                if threshold_proximity <= 1.10:  # 如果在阈值110%以内，继续尝试
                    old_step = step
                    step = max(1, step // 2)
                    if step < old_step:
                        logger.info(f"【激进模式】已接近显存限制，将步长从{old_step}减至{step}进行精细搜索")
                        current_bs = best_bs + step
                        continue
                
                break
            else:
                best_bs = current_bs
                last_successful_est = est_with_overhead
                
                # 动态调整步长 - 激进版本
                memory_usage_ratio = est_with_overhead / (min_gpu_memory * memory_threshold)
                
                if memory_usage_ratio > 0.9:  # 接近内存限制
                    # 当非常接近限制时才减小步长
                    new_step = max(1, step // 2)
                    if new_step < step:
                        step = new_step
                        logger.info(f"【激进模式】已接近显存限制 ({memory_usage_ratio*100:.1f}%)，将步长减至{step}")
                elif current_bs >= 32:
                    # 激进模式使用更大步长增量
                    if aggressive_mode:
                        step = min(step * 2, 16)  # 最大步长16，更快增长
                    else:
                        step = min(step * 2, 8)   # 标准最大步长8
                
                current_bs += step
                
        except Exception as e:
            logger.warning(f"【激进模式】批次大小={current_bs}测试失败: {str(e)}")
            import traceback
            logger.debug(f"异常详情: {traceback.format_exc()}")
            
            # 如果失败但允许重试，则重试
            if retries_left > 0:
                retries_left -= 1
                logger.info(f"【激进模式】剩余{retries_left}次重试机会，降低步长后重试")
                
                # 出错时尝试降低步长继续
                if step > 1:
                    step = step // 2
                    current_bs = best_bs + step
                    logger.info(f"【激进模式】将步长减至{step}并继续")
                    continue
            
            # 重试次数用尽，中断搜索        
            break

    # 设置batch_size并确保是GPU数量的倍数(便于均匀分配)
    # 激进模式下自动向上取整到下一个GPU倍数
    if aggressive_mode and best_bs > initial_bs:
        # 向上取整到下一个GPU倍数
        remainder = best_bs % gpu_count
        if remainder > 0:
            best_bs = best_bs + (gpu_count - remainder)
    else:
        # 标准模式下向下取整到GPU倍数
        best_bs = max((best_bs // gpu_count) * gpu_count, gpu_count)
        
    # 确保批次大小至少为初始值
    if best_bs < initial_bs:
        best_bs = initial_bs
    
    # 添加详细日志
    final_per_gpu_bs = best_bs // gpu_count
    logger.info(f"【激进模式】显存效率分析: 估计使用 {last_successful_est:.2f}MB，显存阈值 {min_gpu_memory * memory_threshold:.2f}MB ({last_successful_est/(min_gpu_memory * memory_threshold)*100:.1f}%)")
    logger.info(f"【激进模式】每个GPU将处理 {final_per_gpu_bs} 个样本/批次")
    
    config['batch_size'] = best_bs
    logger.info(f"【激进模式】确定最终批次大小：总批次={best_bs}，每GPU批次={final_per_gpu_bs}，GPU数量={gpu_count}")
    return config

def estimate_model_memory_per_gpu(model, device, input_size, aggressive_mode=True):
    """
    估算单个GPU上模型前向+反向传播的显存占用 (MB) - 激进版本
    增强版本：
    1. 更精确估算模型参数、梯度和优化器状态的内存占用
    2. 更激进的内存估计，减小安全裕度和保守系数
    3. 优化激活值内存分析，避免过度保守
    4. 提供激进模式选项，允许使用较低的安全裕度
    
    Args:
        model: 模型实例
        device: 计算设备
        input_size: 输入大小 (元组或嵌套元组)
        aggressive_mode: 是否启用激进模式，使用更低的安全裕度
        
    Returns:
        估算的显存占用(MB)
    """
    # 获取原始模型（如果是DataParallel或DDP包装的）
    if hasattr(model, 'module'):
        model_for_estimate = model.module
    else:
        model_for_estimate = model
        
    # 记录初始内存状态
    torch.cuda.empty_cache()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        initial_mem = torch.cuda.memory_allocated(device)
    
    # 确保模型处于训练模式
    model_for_estimate = model_for_estimate.to(device)
    model_for_estimate.train()
    
    # 初始化变量以避免作用域问题
    sample_input = None
    sample_input_1 = None
    sample_input_2 = None
    outputs = None
    forward_mem = 0
    backward_mem = 0
    
    try:
        # 根据输入形状构建示例输入
        if isinstance(input_size, tuple) and len(input_size) == 2 and isinstance(input_size[0], tuple) and isinstance(input_size[1], tuple):
            # 双输入情况（如增强模式）
            sample_input_1 = torch.randn(input_size[0], device=device)
            sample_input_2 = torch.randn(input_size[1], device=device)
            sample_input = (sample_input_1, sample_input_2)
        else:
            # 单输入情况
            sample_input = torch.randn(input_size, device=device)
        
        # 使用混合精度前向传播（如训练时一样）
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            outputs = model_for_estimate(sample_input)
        
        # 测量前向传播内存使用
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            forward_mem = torch.cuda.max_memory_allocated(device) - initial_mem
        
        # 准备反向传播
        # 确定批次大小
        if isinstance(sample_input, tuple):
            bs = sample_input[0].size(0)
        else:
            bs = sample_input.size(0)
        
        # 确定输出类别数
        mod = model_for_estimate
        if hasattr(mod, 'output_layer'):
            num_classes = mod.output_layer.out_features
        elif hasattr(mod, 'cls_head'):
            num_classes = mod.cls_head.out_features
        elif isinstance(outputs, tuple):
            num_classes = outputs[0].size(-1)
        else:
            num_classes = outputs.size(-1)
        
        # 创建随机目标并计算损失
        target = torch.randint(0, num_classes, (bs,), device=device)
        
        # 根据输出形状选择合适的损失函数
        if isinstance(outputs, tuple):
            # 处理多输出的情况（如检测模型）
            loss = nn.CrossEntropyLoss()(outputs[0], target)
        else:
            loss = nn.CrossEntropyLoss()(outputs, target)
        
        # 反向传播
        loss.backward()
        
        # 测量反向传播额外内存使用
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            current_mem = torch.cuda.max_memory_allocated(device)
            backward_mem = current_mem - initial_mem - forward_mem
        
        # 计算内存使用（MB）
        if device.type == 'cuda':
            forward_mb = forward_mem / (1024**2)
            backward_mb = backward_mem / (1024**2)
            
            # 激进模式优化：考虑到PyTorch的内存重用机制，实际激活值内存往往比简单累加更小
            # 正向传播和反向传播的内存有部分重叠
            if aggressive_mode:
                # 激进模式使用更低的累加系数
                activation_mb = forward_mb + backward_mb * 0.85  # 假设15%的内存可重用
            else:
                activation_mb = forward_mb + backward_mb * 0.95  # 标准模式假设5%可重用
            
            # 详细分析模型参数内存
            param_elements = sum(p.numel() for p in model_for_estimate.parameters())
            grad_elements = sum(p.numel() for p in model_for_estimate.parameters() if p.requires_grad)
            
            # 每个parameter占用: 参数值(fp32) + 梯度(fp32) 
            param_mb = param_elements * 4 / (1024**2)  # 参数值: float32 (4字节)
            grad_mb = grad_elements * 4 / (1024**2)    # 梯度: float32 (4字节)
            
            # 优化器状态内存 - 考虑到使用Adam (更精确但仍然激进的估计)
            # 激进模式下假设优化器状态更紧凑
            if aggressive_mode:
                # 激进模式: 假设优化器可以优化内存使用
                optim_mb = grad_elements * 4 * 2.5 / (1024**2)  # 优化器状态: 假设2.5倍梯度大小
            else:
                # 标准模式: 使用标准假设
                optim_mb = grad_elements * 4 * 3 / (1024**2)  # 优化器状态: 假设3倍梯度大小
            
            # 其他运行时内存开销 - 激进模式下降低估计
            if aggressive_mode:
                runtime_mb = (activation_mb + param_mb + grad_mb + optim_mb) * 0.10  # 降至10%
            else:
                runtime_mb = (activation_mb + param_mb + grad_mb + optim_mb) * 0.12  # 降至12%
            
            # 多GPU通信缓冲区 - 激进模式下降低估计
            if hasattr(model, 'module'):  # 如果是多GPU模型
                if aggressive_mode:
                    comm_buffer_mb = grad_mb * 1.8  # 通信缓冲区: 降低估计(1.8倍)
                else:
                    comm_buffer_mb = grad_mb * 2.0  # 标准通信估计(2倍)
            else:
                comm_buffer_mb = 0
            
            # 总内存使用，考虑安全裕度
            # 激进模式使用更小的安全裕度
            total_before_margin = activation_mb + param_mb + grad_mb + optim_mb + runtime_mb + comm_buffer_mb
            
            # 动态安全裕度: 激进模式下使用更小系数
            if aggressive_mode:
                # 激进模式安全系数
                if total_before_margin > 10 * 1024:  # 超过10GB
                    safety_margin = 1.15  # 大模型降至1.15
                elif total_before_margin > 5 * 1024:  # 超过5GB
                    safety_margin = 1.10  # 中模型降至1.10
                else:
                    safety_margin = 1.08  # 小模型降至1.08
            else:
                # 标准模式安全系数
                if total_before_margin > 10 * 1024:  # 超过10GB
                    safety_margin = 1.25  # 大模型1.25
                elif total_before_margin > 5 * 1024:  # 超过5GB
                    safety_margin = 1.20  # 中模型1.20
                else:
                    safety_margin = 1.15  # 小模型1.15
                
            total_mb = total_before_margin * safety_margin
            
            # 创建内存分析报告
            memory_report = {
                'forward_mb': forward_mb,
                'backward_mb': backward_mb,
                'activation_mb': activation_mb,
                'param_mb': param_mb,
                'grad_mb': grad_mb,
                'optim_mb': optim_mb,
                'runtime_mb': runtime_mb,
                'comm_buffer_mb': comm_buffer_mb,
                'total_before_margin_mb': total_before_margin,
                'safety_margin': safety_margin,
                'total_mb': total_mb
            }
            
            # print(f"内存分析报告: {memory_report}")  # 可选的调试输出
        else:
            # CPU模式下简化计算
            param_elements = sum(p.numel() for p in model_for_estimate.parameters())
            grad_elements = sum(p.numel() for p in model_for_estimate.parameters() if p.requires_grad)
            
            param_mb = param_elements * 4 / (1024**2)
            grad_mb = grad_elements * 4 / (1024**2)
            
            # 激进模式下降低CPU内存估计
            if aggressive_mode:
                optim_mb = grad_mb * 2.0  # 假设2.0倍梯度大小
                total_mb = (param_mb + grad_mb + optim_mb) * 1.05  # 更小安全裕度
            else:
                optim_mb = grad_mb * 2.5  # 假设2.5倍梯度大小
                total_mb = (param_mb + grad_mb + optim_mb) * 1.15  # 标准安全裕度
    
    except Exception as e:
        # 捕获异常，以便清理内存
        print(f"内存估算错误: {e}")
        import traceback
        print(traceback.format_exc())
        
        # 简单估计：参数+梯度+优化器状态
        param_elements = sum(p.numel() for p in model_for_estimate.parameters())
        grad_elements = sum(p.numel() for p in model_for_estimate.parameters() if p.requires_grad)
        
        param_mb = param_elements * 4 / (1024**2)
        grad_mb = grad_elements * 4 / (1024**2)
        
        # 激进模式下使用更小的替代安全裕度
        if aggressive_mode:
            optim_mb = grad_mb * 2.0  # 假设优化器状态是梯度的2倍
            total_mb = (param_mb + grad_mb + optim_mb) * 1.5  # 更小的替代安全裕度
        else:
            optim_mb = grad_mb * 2.5  # 假设优化器状态是梯度的2.5倍
            total_mb = (param_mb + grad_mb + optim_mb) * 1.8  # 标准替代安全裕度
    
    finally:
        # 清理内存
        if outputs is not None:
            del outputs
        if sample_input is not None:
            del sample_input
        if sample_input_1 is not None:
            del sample_input_1
        if sample_input_2 is not None:
            del sample_input_2
        
        # 确保所有梯度被清理
        model_for_estimate.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    
    return total_mb

def log_gpu_memory_usage(logger, prefix=None, language='en'):
    """
    记录所有GPU的当前内存使用情况
    
    Args:
        logger: 日志记录器
        prefix: 日志前缀（可选）
        language: 语言设置
    """
    if prefix is None:
        prefix = get_message(language, "gpu_memory_usage_prefix")
        
    if not torch.cuda.is_available():
        logger.info(f"{prefix}: {get_message(language, 'no_gpu_detected')}")
        return
        
    gpu_count = torch.cuda.device_count()
    logger.info(f"{prefix}: {get_message(language, 'using_multi_gpu').format(gpu_count)}")
    
    for i in range(gpu_count):
        try:
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            max_allocated = torch.cuda.max_memory_allocated(i) / (1024**3)
            
            logger.info(f"GPU {i}: {get_message(language, 'gpu_memory_usage_details').format(allocated, reserved, max_allocated)}")
        except Exception as e:
            logger.warning(get_message(language, "cannot_get_gpu_memory").format(i, e))
