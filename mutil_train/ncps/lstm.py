# -*- coding: utf-8 -*-
# train/ncps/lstm.py
# LSTM单元实现，可用于混合记忆模式

import torch
from torch import nn

class LSTMCell(nn.Module):
    """
    LSTM单元实现
    用于混合记忆模式，帮助处理长期依赖关系
    """
    def __init__(self, input_size, hidden_size):
        """
        初始化LSTM单元
        
        Args:
            input_size: 输入特征数量
            hidden_size: 隐藏状态大小
        """
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # LSTM门控参数
        self.input_map = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.recurrent_map = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        # 初始化权重
        self.init_weights()

    def init_weights(self):
        """权重初始化，使用均匀分布和正交初始化"""
        for w in self.input_map.parameters():
            if w.dim() == 1:
                torch.nn.init.uniform_(w, -0.1, 0.1)
            else:
                torch.nn.init.xavier_uniform_(w)
        for w in self.recurrent_map.parameters():
            if w.dim() == 1:
                torch.nn.init.uniform_(w, -0.1, 0.1)
            else:
                torch.nn.init.orthogonal_(w)

    def forward(self, inputs, states):
        """
        前向传播函数
        
        Args:
            inputs: 输入张量 [B, input_size]
            states: 状态元组 (output_state, cell_state)
            
        Returns:
            (output_state, cell_state): 更新后的状态
        """
        # 确保所有输入在同一设备上
        device = inputs.device
        output_state, cell_state = states
        
        if output_state.device != device:
            output_state = output_state.to(device)
        if cell_state.device != device:
            cell_state = cell_state.to(device)
        
        # LSTM门控机制
        z = self.input_map(inputs) + self.recurrent_map(output_state)
        i, ig, fg, og = z.chunk(4, 1)

        input_activation = self.tanh(i)
        input_gate = self.sigmoid(ig)
        forget_gate = self.sigmoid(fg + 1.0)  # +1.0为遗忘门添加偏置，改善长期记忆
        output_gate = self.sigmoid(og)

        # 更新单元状态和输出状态
        new_cell = cell_state * forget_gate + input_activation * input_gate
        output_state = self.tanh(new_cell) * output_gate

        return output_state, new_cell