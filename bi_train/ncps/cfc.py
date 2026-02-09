# -*- coding: utf-8 -*-
# train/ncps/cfc.py
# CfC模块实现，针对多GPU训练进行了优化

import torch
from torch import nn
from typing import Optional, Union

from .cfc_cell import CfCCell
from .wired_cfc_cell import WiredCfCCell
from .lstm import LSTMCell

class CfC(nn.Module):
    """
    闭式连续时间 (Closed-form Continuous-time) RNN模块
    支持多GPU训练和自动处理设备迁移
    """
    def __init__(
        self,
        input_size,
        units,
        proj_size: Optional[int] = None,
        return_sequences: bool = True,
        batch_first: bool = True,
        mixed_memory: bool = False,
        mode: str = "default",
        activation: str = "lecun_tanh",
        backbone_units: Optional[int] = None,
        backbone_layers: Optional[int] = None,
        backbone_dropout: Optional[int] = None,
    ):
        """
        应用闭式连续时间RNN到输入序列
        
        Args:
            input_size: 输入特征数量或连接结构
            units: 隐藏单元数量
            proj_size: 输出投影维度(可选)
            return_sequences: 是否返回完整序列还是仅最后输出
            batch_first: 批次维度是否为第一维
            mixed_memory: 是否使用混合记忆增强RNN
            mode: 模式："default", "pure", "no_gate"
            activation: 骨干网络的激活函数
            backbone_units: 骨干网络隐藏单元数量(默认128)
            backbone_layers: 骨干网络层数(默认1)
            backbone_dropout: 骨干网络dropout比例(默认0)
        """
        super(CfC, self).__init__()
        self.input_size = input_size
        self.wiring_or_units = units
        self.proj_size = proj_size
        self.batch_first = batch_first
        self.return_sequences = return_sequences

        # 检查是否使用连接结构(wiring)
        if hasattr(units, 'build') and callable(getattr(units, 'build')):
            self.wired_mode = True
            if backbone_units is not None:
                raise ValueError(f"在连接模式下不能使用backbone_units")
            if backbone_layers is not None:
                raise ValueError(f"在连接模式下不能使用backbone_layers")
            if backbone_dropout is not None:
                raise ValueError(f"在连接模式下不能使用backbone_dropout")
                
            # 设置连接结构
            self.wiring = units
            self.state_size = self.wiring.units
            self.output_size = self.wiring.output_dim
            self.rnn_cell = WiredCfCCell(
                input_size,
                self.wiring_or_units,
                mode,
            )
        else:
            self.wired_false = True
            backbone_units = 128 if backbone_units is None else backbone_units
            backbone_layers = 1 if backbone_layers is None else backbone_layers
            backbone_dropout = 0.0 if backbone_dropout is None else backbone_dropout
            self.state_size = units
            self.output_size = self.state_size
            self.rnn_cell = CfCCell(
                input_size,
                self.wiring_or_units,
                mode,
                activation,
                backbone_units,
                backbone_layers,
                backbone_dropout,
            )
            
        self.use_mixed = mixed_memory
        if self.use_mixed:
            self.lstm = LSTMCell(input_size, self.state_size)

        # 输出投影层
        if proj_size is None:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(self.output_size, self.proj_size)

    def forward(self, input, hx=None, timespans=None):
        """
        前向传播函数
        
        Args:
            input: 输入张量 [L,C](无批次)或[B,L,C](batch_first=True)或[L,B,C](batch_first=False)
            hx: 初始隐藏状态 [B,H](mixed_memory=False)或((B,H),(B,H))(mixed_memory=True)
            timespans: 时间跨度
            
        Returns:
            (output, hx): 输出和最终隐藏状态
        """
        device = input.device
        is_batched = input.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0
        
        # 处理无批次输入
        if not is_batched:
            input = input.unsqueeze(batch_dim)
            if timespans is not None:
                timespans = timespans.unsqueeze(batch_dim)

        batch_size, seq_len = input.size(batch_dim), input.size(seq_dim)

        # 初始化隐藏状态
        if hx is None:
            h_state = torch.zeros((batch_size, self.state_size), device=device)
            c_state = (
                torch.zeros((batch_size, self.state_size), device=device)
                if self.use_mixed
                else None
            )
        else:
            # 验证隐藏状态格式
            if self.use_mixed and isinstance(hx, torch.Tensor):
                raise RuntimeError(
                    "使用mixed_memory=True运行CfC时，需要传入元组(h0,c0)作为状态(收到torch.Tensor)"
                )
                
            h_state, c_state = hx if self.use_mixed else (hx, None)
            
            # 检查批次维度
            if is_batched:
                if h_state.dim() != 2:
                    msg = (
                        "对于批次化2D输入，hx和cx也应该"
                        f"是2D张量，但收到({h_state.dim()}-D)张量"
                    )
                    raise RuntimeError(msg)
            else:
                # 无批次模式
                if h_state.dim() != 1:
                    msg = (
                        "对于非批次化1D输入，hx和cx也应该"
                        f"是1D张量，但收到({h_state.dim()}-D)张量"
                    )
                    raise RuntimeError(msg)
                h_state = h_state.unsqueeze(0)
                c_state = c_state.unsqueeze(0) if c_state is not None else None

        # 确保隐藏状态在正确设备上
        h_state = h_state.to(device)
        if c_state is not None:
            c_state = c_state.to(device)

        # 循环处理序列
        output_sequence = []
        for t in range(seq_len):
            # 根据batch_first获取当前时间步
            if self.batch_first:
                inputs = input[:, t]
                ts = 1.0 if timespans is None else timespans[:, t].squeeze()
            else:
                inputs = input[t]
                ts = 1.0 if timespans is None else timespans[t].squeeze()

            # 运行细胞单元
            if self.use_mixed:
                h_state, c_state = self.lstm(inputs, (h_state, c_state))
            h_out, h_state = self.rnn_cell.forward(inputs, h_state, ts)
            
            # 收集输出序列
            if self.return_sequences:
                output_sequence.append(self.fc(h_out))

        # 准备返回值
        if self.return_sequences:
            stack_dim = 1 if self.batch_first else 0
            readout = torch.stack(output_sequence, dim=stack_dim)
        else:
            readout = self.fc(h_out)
        hx = (h_state, c_state) if self.use_mixed else h_state

        # 处理无批次输出
        if not is_batched:
            readout = readout.squeeze(batch_dim)
            hx = (h_state[0], c_state[0]) if self.use_mixed else h_state[0]

        return readout, hx