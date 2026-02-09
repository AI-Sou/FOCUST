# -*- coding: utf-8 -*-

#ncps/cfc_wrapper.py
# train/ncps/wrapper.py - 完全重构版
# 优化CfC包装类，解决多GPU训练设备不一致问题

import torch
from torch import nn
import os
import sys

# 添加当前路径到sys.path，确保可以导入train.ncps包
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from .cfc import CfC
from .wirings import AutoNCP

def lecun_tanh(x: torch.Tensor) -> torch.Tensor:
    """
    LeCun Tanh 激活函数定义
    固定放大比例的Tanh，有利于稳定训练
    """
    return 1.7159 * torch.tanh(0.666 * x)

class CfCWrapper(nn.Module):
    """
    改进的CfC包装类，完全兼容多GPU训练
    """
    # !! 核心修复：在__init__函数签名中添加 batch_first 参数 !!
    def __init__(self, input_size, units, activation=None, return_sequences=True, batch_first=True):
        """
        初始化CfC包装器。

        Args:
            input_size (int): 输入特征维度。
            units (Wiring): NCP连接结构。
            activation (callable, optional): 激活函数。
            return_sequences (bool, optional): 是否返回完整序列。
            batch_first (bool, optional): 输入张量的第一维是否为批次大小。
        """
        super(CfCWrapper, self).__init__()
        
        # 创建标准CfC模块，并将 batch_first 参数传递给它
        self.cfc = CfC(
            input_size=input_size,
            units=units,
            mode="default",
            activation=activation if activation else lecun_tanh,
            return_sequences=return_sequences,
            batch_first=batch_first # !! 核心修复：将参数传递给真正的CfC类 !!
        )
        
        # 关键改进：确保初始化时所有sparsity_mask被正确创建为buffer
        self._ensure_sparsity_masks_are_buffers()
        
    def _ensure_sparsity_masks_are_buffers(self):
        """
        关键方法：确保所有sparsity_mask是正确的buffer
        """
        # 递归地查找并修复模型中的所有sparsity_mask
        def fix_masks(module):
            if hasattr(module, 'sparsity_mask') and module.sparsity_mask is not None:
                # 检查是否已经是buffer
                is_buffer = any(name == 'sparsity_mask' for name, _ in module.named_buffers())
                
                # 如果不是buffer，重新注册
                if not is_buffer:
                    mask_data = module.sparsity_mask.clone()
                    # 确保删除的是属性而不是变量
                    if hasattr(module, 'sparsity_mask'):
                        delattr(module, 'sparsity_mask')
                    module.register_buffer('sparsity_mask', mask_data)
            
            # 递归地对子模块进行操作
            for child in module.children():
                fix_masks(child)

        fix_masks(self.cfc)
    
    def forward(self, x, h0=None, timespans=None):
        """
        简化的前向传播方法。
        """
        if timespans is not None and not isinstance(timespans, (int, float)):
            timespans = timespans.to(x.device)
        
        if h0 is not None:
            if isinstance(h0, tuple):
                h0 = tuple(h.to(x.device) for h in h0)
            else:
                h0 = h0.to(x.device)
        
        return self.cfc(x, h0, timespans)

    @property
    def state_size(self):
        """获取状态大小"""
        return self.cfc.state_size if hasattr(self.cfc, 'state_size') else None
        
    @property
    def output_size(self):
        """获取输出大小"""
        return self.cfc.output_size if hasattr(self.cfc, 'output_size') else None