# -*- coding: utf-8 -*-
# train/ncps/wired_cfc_cell.py
# 修复了类定义和导入问题

import numpy as np
import torch
from torch import nn

from .cfc_cell import CfCCell


class WiredCfCCell(nn.Module):
    """
    有连接结构的CfC单元，实现层次化神经连接
    优化版本：确保多GPU训练时所有张量在同一设备上
    """
    def __init__(
        self,
        input_size,
        wiring,
        mode="default",
    ):
        """
        初始化有连接结构的CfC单元
        
        Args:
            input_size: 输入特征数量
            wiring: 神经连接结构
            mode: 运行模式 "default", "pure", "no_gate"
        """
        super(WiredCfCCell, self).__init__()

        # 构建连接结构
        if input_size is not None:
            wiring.build(input_size)
        if not wiring.is_built():
            raise ValueError(
                "连接错误！未知的输入特征数量。请传递参数'input_size'或调用'wiring.build()'。"
            )
        self._wiring = wiring

        # 创建多层CfC
        self._layers = nn.ModuleList()  # 使用ModuleList确保正确注册
        in_features = wiring.input_dim
        for l in range(wiring.num_layers):
            # 获取当前层的神经元
            hidden_units = self._wiring.get_neurons_of_layer(l)
            
            # 构建输入稀疏连接矩阵
            if l == 0:
                # 第一层：感觉输入到隐藏单元
                input_sparsity = self._wiring.sensory_adjacency_matrix[:, hidden_units]
            else:
                # 后续层：上一层到当前层的连接
                prev_layer_neurons = self._wiring.get_neurons_of_layer(l - 1)
                input_sparsity = self._wiring.adjacency_matrix[:, hidden_units]
                input_sparsity = input_sparsity[prev_layer_neurons, :]
                
            # 添加自连接（用于跳跃连接）
            input_sparsity = np.concatenate(
                [
                    input_sparsity,
                    np.ones((len(hidden_units), len(hidden_units))),
                ],
                axis=0,
            )

            # 创建CfC单元
            # 使用ModuleList替代register_module
            rnn_cell = CfCCell(
                in_features,
                len(hidden_units),
                mode,
                backbone_activation="lecun_tanh",
                backbone_units=0,
                backbone_layers=0,
                backbone_dropout=0.0,
                sparsity_mask=input_sparsity,
            )
            self._layers.append(rnn_cell)
            in_features = len(hidden_units)

    @property
    def state_size(self):
        """获取状态大小"""
        return self._wiring.units

    @property
    def layer_sizes(self):
        """获取每层大小"""
        return [
            len(self._wiring.get_neurons_of_layer(i))
            for i in range(self._wiring.num_layers)
        ]

    @property
    def num_layers(self):
        """获取层数"""
        return self._wiring.num_layers

    @property
    def sensory_size(self):
        """获取感觉输入大小"""
        return self._wiring.input_dim

    @property
    def motor_size(self):
        """获取电机输出大小"""
        return self._wiring.output_dim

    @property
    def output_size(self):
        """获取输出大小"""
        return self.motor_size

    @property
    def synapse_count(self):
        """获取突触数量"""
        return np.sum(np.abs(self._wiring.adjacency_matrix))

    @property
    def sensory_synapse_count(self):
        """获取感觉突触数量"""
        return np.sum(np.abs(self._wiring.adjacency_matrix))

    def forward(self, input, hx, timespans):
        """
        前向传播函数
        
        Args:
            input: 输入张量 [B, input_size]
            hx: 隐藏状态 [B, state_size]
            timespans: 时间跨度
            
        Returns:
            (h, new_h_state): 输出和新的隐藏状态
        """
        # 确保所有输入在同一设备上
        device = input.device
        if hx.device != device:
            hx = hx.to(device)
        if isinstance(timespans, torch.Tensor) and timespans.device != device:
            timespans = timespans.to(device)
            
        # 将隐藏状态分割到各层
        h_state = torch.split(hx, self.layer_sizes, dim=1)

        # 逐层前向传播
        new_h_state = []
        inputs = input
        
        for i in range(self.num_layers):
            # 确保当前层的隐藏状态在正确设备上
            layer_h = h_state[i].to(device)
            
            # 使用当前层的CfC单元处理
            h, _ = self._layers[i].forward(inputs, layer_h, timespans)
            inputs = h
            new_h_state.append(h)

        # 拼接所有层的新隐藏状态
        new_h_state = torch.cat(new_h_state, dim=1)
        
        return h, new_h_state