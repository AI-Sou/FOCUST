# -*- coding: utf-8 -*-
# train/ncps/cfc_cell.py
# CfC单元实现（经过改进以支持多GPU训练）

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Optional, Union

class LeCun(nn.Module):
    """
    LeCun Tanh激活函数
    有特定的缩放系数，有利于稳定训练
    """
    def __init__(self):
        super(LeCun, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)


class CfCCell(nn.Module):
    """
    闭式连续时间(Closed-form Continuous-time)细胞单元
    优化版本：包含设备一致性处理，确保在多GPU环境下正常工作
    """
    def __init__(
        self,
        input_size,
        hidden_size,
        mode="default",
        backbone_activation="lecun_tanh",
        backbone_units=128,
        backbone_layers=1,
        backbone_dropout=0.0,
        sparsity_mask=None,
    ):
        """
        构建CfC单元模块
        
        Args:
            input_size: 输入特征大小
            hidden_size: 隐藏状态大小
            mode: "default", "pure", "no_gate" 之一
            backbone_activation: 骨干网络的激活函数
            backbone_units: 骨干网络的隐藏单元数量
            backbone_layers: 骨干网络的层数
            backbone_dropout: 骨干网络的dropout比例
            sparsity_mask: 稀疏连接掩码
        """
        super(CfCCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        allowed_modes = ["default", "pure", "no_gate"]
        if mode not in allowed_modes:
            raise ValueError(
                f"未知模式'{mode}'，有效选项为{str(allowed_modes)}"
            )
            
        # 关键改进：确保sparsity_mask始终是register_buffer而不是parameter
        if sparsity_mask is not None:
            self.register_buffer('sparsity_mask', 
                               torch.from_numpy(np.abs(sparsity_mask.T).astype(np.float32)))
        else:
            self.sparsity_mask = None

        self.mode = mode

        # 设置激活函数
        if backbone_activation == "silu":
            backbone_activation = nn.SiLU
        elif backbone_activation == "relu":
            backbone_activation = nn.ReLU
        elif backbone_activation == "tanh":
            backbone_activation = nn.Tanh
        elif backbone_activation == "gelu":
            backbone_activation = nn.GELU
        elif backbone_activation == "lecun_tanh":
            backbone_activation = LeCun
        else:
            raise ValueError(f"未知激活函数 {backbone_activation}")

        # 构建骨干网络
        self.backbone = None
        self.backbone_layers = backbone_layers
        if backbone_layers > 0:
            layer_list = [
                nn.Linear(input_size + hidden_size, backbone_units),
                backbone_activation(),
            ]
            for i in range(1, backbone_layers):
                layer_list.append(nn.Linear(backbone_units, backbone_units))
                layer_list.append(backbone_activation())
                if backbone_dropout > 0.0:
                    layer_list.append(torch.nn.Dropout(backbone_dropout))
            self.backbone = nn.Sequential(*layer_list)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        cat_shape = int(
            self.hidden_size + input_size if backbone_layers == 0 else backbone_units
        )

        # 主要网络层
        self.ff1 = nn.Linear(cat_shape, hidden_size)
        if self.mode == "pure":
            self.w_tau = torch.nn.Parameter(
                data=torch.zeros(1, self.hidden_size), requires_grad=True
            )
            self.A = torch.nn.Parameter(
                data=torch.ones(1, self.hidden_size), requires_grad=True
            )
        else:
            self.ff2 = nn.Linear(cat_shape, hidden_size)
            self.time_a = nn.Linear(cat_shape, hidden_size)
            self.time_b = nn.Linear(cat_shape, hidden_size)
        self.init_weights()

    def init_weights(self):
        """初始化网络权重"""
        for w in self.parameters():
            if w.dim() == 2 and w.requires_grad:
                torch.nn.init.xavier_uniform_(w)

    def forward(self, input, hx, ts):
        """
        前向传播函数
        
        Args:
            input: 输入张量 [B, input_size]
            hx: 隐藏状态 [B, hidden_size]
            ts: 时间跨度
            
        Returns:
            new_hidden: 更新后的隐藏状态
        """
        # 确保输入在同一设备上
        device = input.device
        if hx.device != device:
            hx = hx.to(device)
        if isinstance(ts, torch.Tensor) and ts.device != device:
            ts = ts.to(device)
            
        x = torch.cat([input, hx], 1)
        if self.backbone_layers > 0:
            x = self.backbone(x)
            
        # 使用稀疏掩码进行线性变换
        if self.sparsity_mask is not None:
            # 确保掩码在同一设备上
            ff1 = F.linear(x, self.ff1.weight * self.sparsity_mask, self.ff1.bias)
        else:
            ff1 = self.ff1(x)
            
        if self.mode == "pure":
            # 解析解模式
            # 关键改进：确保所有张量在同一设备上
            w_tau = self.w_tau.to(device)
            A = self.A.to(device)
            
            # 计算新的隐藏状态
            new_hidden = (
                -A
                * torch.exp(-ts * (torch.abs(w_tau) + torch.abs(ff1)))
                * ff1
                + A
            )
        else:
            # 标准CfC模式
            if self.sparsity_mask is not None:
                # 确保掩码在同一设备上
                ff2 = F.linear(x, self.ff2.weight * self.sparsity_mask, self.ff2.bias)
            else:
                ff2 = self.ff2(x)
                
            ff1 = self.tanh(ff1)
            ff2 = self.tanh(ff2)
            t_a = self.time_a(x)
            t_b = self.time_b(x)
            t_interp = self.sigmoid(t_a * ts + t_b)
            
            if self.mode == "no_gate":
                new_hidden = ff1 + t_interp * ff2
            else:
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2
                
        return new_hidden, new_hidden