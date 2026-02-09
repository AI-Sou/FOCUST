# 目标检测模型模块 - 处理大图像(4000x4000)的VeritasOD模型实现

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import torch.utils.checkpoint as checkpoint
try:
    from .config_utils import get_message
    from ..ncps.wrapper import CfCWrapper, lecun_tanh
    from ..ncps.wirings import AutoNCP
except ImportError:  # pragma: no cover
    from train.config_utils import get_message
    try:
        from bi_train.ncps.wrapper import CfCWrapper, lecun_tanh
        from bi_train.ncps.wirings import AutoNCP
    except ImportError:
        from ncps.wrapper import CfCWrapper, lecun_tanh
        from ncps.wirings import AutoNCP

class EffcientDepthwiseSeparableConv(nn.Module):
    """
    优化的深度可分离卷积，减少参数量和显存占用
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(EffcientDepthwiseSeparableConv, self).__init__()
        # 深度卷积层 - 用于空间特征提取
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias
        )
        # 逐点卷积层 - 用于通道映射
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )
        # 初始化权重
        self._init_weight()

    def _init_weight(self):
        """高效初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """使用原地操作减少内存使用"""
        # 深度卷积 - 空间特征提取
        x = self.depthwise(x)
        # 逐点卷积 - 通道映射
        x = self.pointwise(x)
        return x


class StreamTiledImageProcessor(nn.Module):
    """
    流式分块图像处理器，专为高分辨率图像优化
    仅用于目标检测模型，不用于分类模型
    """
    def __init__(self, base_processor, tile_size=1024, overlap_ratio=0.25, 
                 max_batch_size=1, dynamic_tile_size=True):
        super(StreamTiledImageProcessor, self).__init__()
        self.base_processor = base_processor
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.max_batch_size = max_batch_size  # 每批最多处理的图块数
        self.dynamic_tile_size = dynamic_tile_size  # 是否动态调整图块大小
        
    def _get_optimal_tile_size(self, image_size, free_memory_mb):
        """动态计算最佳图块大小，基于可用显存"""
        # 默认图块大小
        suggested_size = self.tile_size
        
        # 根据可用显存动态调整
        if self.dynamic_tile_size:
            # 假设每像素占用约16字节(RGBA float32)
            bytes_per_pixel = 16
            # 计算在可用显存下可以安全处理的图块大小
            mem_in_bytes = free_memory_mb * 1024 * 1024
            max_pixels = mem_in_bytes / bytes_per_pixel / 4  # 预留75%内存给中间结果
            max_tile_size = int(math.sqrt(max_pixels))
            
            # 最终图块大小：默认值和计算值中的较小值
            suggested_size = min(self.tile_size, max_tile_size)
            # 确保至少为256，且为32的倍数（对GPU友好）
            suggested_size = max(256, (suggested_size // 32) * 32)
        
        return suggested_size
        
    def forward(self, x):
        """
        优化的流式分块处理，逐块处理图像以降低显存使用
        
        Args:
            x: 输入图像 [B, C, H, W]
            
        Returns:
            处理后的特征和位置信息列表
        """
        B, C, H, W = x.shape
        device = x.device
        
        # 获取当前可用显存
        available_mem = torch.cuda.get_device_properties(0).total_memory
        allocated_mem = torch.cuda.memory_allocated(0)
        free_mem_mb = (available_mem - allocated_mem) / (1024 * 1024)
        
        # 根据可用显存计算最佳图块大小
        tile_size = self._get_optimal_tile_size(max(H, W), free_mem_mb)
        
        # 如果图像小于图块大小，直接处理整图
        if H <= tile_size and W <= tile_size:
            features = self.base_processor(x)
            return [(features, (0, H, 0, W))]
        
        # 计算重叠大小和步长
        overlap = int(tile_size * self.overlap_ratio)
        stride = tile_size - overlap
        
        # 计算需要的块数
        h_blocks = max(1, (H - overlap) // stride)
        w_blocks = max(1, (W - overlap) // stride)
        
        # 确保覆盖整个图像
        if stride * h_blocks + overlap < H:
            h_blocks += 1
        if stride * w_blocks + overlap < W:
            w_blocks += 1
        
        # 存储结果
        results = []
        
        # 分组处理图块，控制每次处理的数量
        for i in range(0, h_blocks):
            for j in range(0, w_blocks, self.max_batch_size):
                # 确定当前批次要处理的范围
                batch_end = min(j + self.max_batch_size, w_blocks)
                current_tiles = []
                current_positions = []
                
                # 收集当前批次的所有图块
                for jj in range(j, batch_end):
                    # 计算图块坐标
                    h_start = min(i * stride, H - tile_size)
                    w_start = min(jj * stride, W - tile_size)
                    h_start = max(0, h_start)  # 确保不为负
                    w_start = max(0, w_start)  # 确保不为负
                    h_end = min(h_start + tile_size, H)
                    w_end = min(w_start + tile_size, W)
                    
                    # 提取图块
                    tile = x[:, :, h_start:h_end, w_start:w_end]
                    
                    # 处理不足tile_size的边缘情况
                    if tile.shape[2] < tile_size or tile.shape[3] < tile_size:
                        # 创建全零图块
                        padded_tile = torch.zeros((B, C, tile_size, tile_size),
                                               device=device, dtype=x.dtype)
                        # 填充有效区域
                        padded_tile[:, :, :tile.shape[2], :tile.shape[3]] = tile
                        tile = padded_tile
                    
                    current_tiles.append(tile)
                    current_positions.append((h_start, h_end, w_start, w_end))
                
                # 如果只有一个图块，直接处理
                if len(current_tiles) == 1:
                    tile_result = self.base_processor(current_tiles[0])
                    results.append((tile_result, current_positions[0]))
                else:
                    # 批量处理多个图块
                    try:
                        # 合并为一个批次
                        batch_tiles = torch.cat(current_tiles, dim=0)
                        batch_features = self.base_processor(batch_tiles)
                        
                        # 分离每个图块的结果
                        for idx, pos in enumerate(current_positions):
                            # 获取单个图块结果
                            start_idx = idx * B
                            end_idx = start_idx + B
                            tile_result = batch_features[start_idx:end_idx]
                            results.append((tile_result, pos))
                            
                    except RuntimeError as e:
                        # 如果发生内存错误，改为逐个处理
                        if "CUDA out of memory" in str(e):
                            # 逐个处理
                            for idx, (tile, pos) in enumerate(zip(current_tiles, current_positions)):
                                tile_result = self.base_processor(tile)
                                results.append((tile_result, pos))
                        else:
                            # 其他错误则抛出
                            raise
        
        return results


class LargeImageCNNFeatureExtractor(nn.Module):
    """
    大图像CNN特征提取器 - 为目标检测模型优化，支持多尺度特征提取
    """
    def __init__(self, feature_dim=64, num_conv_layers=3, base_channels=32, 
                 image_size=4000, use_checkpointing=False, return_multi_scale=True):
        super(LargeImageCNNFeatureExtractor, self).__init__()
        
        self.image_size = image_size
        self.use_checkpointing = use_checkpointing
        self.return_multi_scale = return_multi_scale
        self.feature_dim = feature_dim
        
        # 初始卷积层 - 减少输入通道 (3->32)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # 构建主干网络
        self.layers = nn.ModuleList()
        in_channels = base_channels
        out_channels = base_channels
        
        # 存储每层输出通道数
        self.out_channels = []
        
        for i in range(num_conv_layers):
            stage = nn.Sequential()
            
            # 使用深度可分离卷积代替标准卷积，减少参数和显存
            conv = EffcientDepthwiseSeparableConv(
                in_channels, out_channels,
                kernel_size=3, stride=1, padding=1, bias=False
            )
            stage.add_module(f'dw_conv{i}', conv)
            stage.add_module(f'bn{i}', nn.BatchNorm2d(out_channels))
            stage.add_module(f'relu{i}', nn.ReLU(inplace=True))
            
            # 仅在部分层使用池化以保留空间信息
            if i % 2 == 1:
                stage.add_module(f'pool{i}', nn.MaxPool2d(2, stride=2))
                
            self.layers.append(stage)
            self.out_channels.append(out_channels)
            
            in_channels = out_channels
            # 更温和地增加通道数
            if i < num_conv_layers - 1:
                out_channels = min(out_channels * 2, 128)  # 限制最大通道数
                
        # 自适应池化和维度映射
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, feature_dim)
        
        # 多尺度特征转换层 (如果启用)
        if self.return_multi_scale:
            self.lateral_convs = nn.ModuleList()
            for out_c in self.out_channels:
                self.lateral_convs.append(
                    nn.Conv2d(out_c, feature_dim, kernel_size=1, bias=False)
                )
                
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """高效初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def _forward_impl(self, x):
        """实际的前向传播实现"""
        # 初始卷积
        x = self.initial_conv(x)
        
        # 如果需要多尺度特征
        if self.return_multi_scale:
            multi_scale_features = []
            
            # 通过每个阶段并保存中间特征
            for i, layer in enumerate(self.layers):
                x = layer(x)
                
                # 保存较深层的特征
                if i >= len(self.layers) // 2:  # 只保存后半部分的特征
                    # 将特征转换为统一的通道数
                    transformed = self.lateral_convs[i](x)
                    multi_scale_features.append(transformed)
            
            # 最终特征处理
            pooled = self.adaptive_pool(x)
            
            x = pooled.reshape(pooled.size(0), -1)
            
            x = self.fc(x)
            
            return x, multi_scale_features
        else:
            # 单尺度处理
            for layer in self.layers:
                x = layer(x)
                
            x = self.adaptive_pool(x)
            
            x = x.reshape(x.size(0), -1)
            
            x = self.fc(x)
            
            return x
        
    def forward(self, x):
        """支持检查点的前向传播"""
        if self.use_checkpointing and self.training:
            return checkpoint.checkpoint(self._forward_impl, x)
        else:
            return self._forward_impl(x)


class OptimizedVeritpModule(nn.Module):
    """
    优化的空间-时间特征处理模块，降低显存占用
    """
    def __init__(self, in_dim, use_checkpointing=False):
        super(OptimizedVeritpModule, self).__init__()
        self.use_checkpointing = use_checkpointing
        self.in_dim = in_dim
        
        # 减少中间特征维度，降低显存使用
        hidden_dim = in_dim // 4
        
        # 空间注意力，使用较小的隐藏层
        self.spatial_attention = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim, bias=False),
            nn.Sigmoid()
        )
        
        # 时间注意力 - 使用1D卷积获取时序依赖
        self.temporal_conv = nn.Conv1d(
            in_dim, in_dim, 
            kernel_size=3, padding=1, 
            groups=in_dim,  # 深度可分离卷积，减少参数
            bias=False
        )
        self.temporal_attention = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim, bias=False),
            nn.Sigmoid()
        )
        
        # 特征调整层使用层归一化而非线性层
        self.layer_norm = nn.LayerNorm(in_dim)
        
    def _forward_impl(self, x):
        """实际的前向传播实现"""
        # 空间注意力 - 关注每个时间步的空间特征
        spatial_weights = self.spatial_attention(x)
        x_spatial = x * spatial_weights
        
        # 时间注意力 - 关注特征的时序变化
        # 调整维度以适应1D操作
        x_t = x.permute(0, 2, 1).contiguous()  # [B, C, T]
        
        # 应用卷积和注意力
        x_t_conv = self.temporal_conv(x_t)
        x_t_conv = x_t_conv.permute(0, 2, 1).contiguous()  # [B, T, C]
        
        temporal_weights = self.temporal_attention(x_t_conv)
        x_temporal = x * temporal_weights
        
        # 融合空间和时间特征，使用残差连接和层归一化
        x_fused = x + x_spatial + x_temporal
        x_adjusted = self.layer_norm(x_fused)
        
        return x_adjusted
    
    def forward(self, x):
        """支持检查点的前向传播"""
        if self.use_checkpointing and self.training:
            return checkpoint.checkpoint(self._forward_impl, x)
        else:
            return self._forward_impl(x)


class OptimizedColonyGrowthExtractor(nn.Module):
    """
    优化的菌落生长特征提取器，降低显存占用
    """
    def __init__(self, feature_dim, use_checkpointing=False):
        super(OptimizedColonyGrowthExtractor, self).__init__()
        self.feature_dim = feature_dim
        self.use_checkpointing = use_checkpointing
        
        # 隐藏维度减小，降低显存占用
        hidden_dim = feature_dim // 2
        
        # 时序差异提取层 - 捕获相邻帧之间的变化
        self.temporal_diff_net = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 最终特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim)
        )
        
    def _forward_impl(self, x):
        """实际的前向传播实现"""
        # 调整维度以适应1D操作
        x_t = x.permute(0, 2, 1).contiguous()  # [B, C, T]
        
        # 计算相邻帧差异
        diff_features = self.temporal_diff_net(x_t)  # [B, C/2, T-1]
        
        # 调整维度并应用融合层
        diff_features = diff_features.permute(0, 2, 1).contiguous()  # [B, T-1, C/2]
        
        # 应用融合层
        enhanced = self.fusion_layer(diff_features)  # [B, T-1, C]
        
        return enhanced
    
    def forward(self, x):
        """支持检查点的前向传播"""
        if self.use_checkpointing and self.training:
            return checkpoint.checkpoint(self._forward_impl, x)
        else:
            return self._forward_impl(x)


class EfficientAttentionFusion(nn.Module):
    """
    优化的注意力双模态融合模块，显著减少显存占用
    """
    def __init__(self, feature_dim, use_checkpointing=False):
        super(EfficientAttentionFusion, self).__init__()
        self.feature_dim = feature_dim
        self.use_checkpointing = use_checkpointing
        
        # 减少中间特征维度，降低显存占用
        self.hidden_dim = feature_dim // 4
        
        # 注意力融合所需的层
        self.query_proj = nn.Linear(feature_dim, self.hidden_dim, bias=False)
        self.key_proj = nn.Linear(feature_dim, self.hidden_dim, bias=False)
        self.value_proj = nn.Linear(feature_dim, self.hidden_dim, bias=False)
        
        # 输出投影
        self.fc_out = nn.Linear(self.hidden_dim, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """高效初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def _attention_fusion(self, x1, x2):
        """优化的注意力融合实现"""
        # 维度投影 - 减少计算量
        q = self.query_proj(x1)
        k = self.key_proj(x2)
        v = self.value_proj(x2)
        
        # 计算注意力得分 (缩放点积)
        scale = math.sqrt(self.hidden_dim)
        attn_scores = torch.bmm(q, k.transpose(-2, -1)) / scale
        
        # 应用softmax得到注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 应用注意力权重
        context = torch.bmm(attn_weights, v)
        
        # 输出投影
        out = self.fc_out(context)
        
        # 残差连接和层归一化
        out = x1 + out
        out = self.layer_norm(out)
        
        return out
    
    def _forward_impl(self, x1, x2):
        """实际的前向传播实现"""
        # 使用注意力机制进行融合
        return self._attention_fusion(x1, x2)
    
    def forward(self, x1, x2):
        """支持检查点的前向传播"""
        if self.use_checkpointing and self.training:
            return checkpoint.checkpoint(self._forward_impl, x1, x2)
        else:
            return self._forward_impl(x1, x2)


class EfficientPathFusion(nn.Module):
    """
    优化的路径融合模块，减少显存占用
    """
    def __init__(self, feature_dim, num_paths=2, language='en', use_checkpointing=False):
        super(EfficientPathFusion, self).__init__()
        self.feature_dim = feature_dim
        self.num_paths = num_paths
        self.language = language
        self.use_checkpointing = use_checkpointing
        
        # 路径注意力机制 - 更轻量级的设计
        self.path_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 输出层归一化
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def _forward_impl(self, features):
        """实际的前向传播实现"""
        # 验证输入特征数量
        assert len(features) == self.num_paths, get_message(self.language, "feature_path_mismatch").format(
            self.num_paths, len(features)
        )
        
        # 为每个特征计算注意力权重
        weights = []
        for feat in features:
            weight = self.path_attention(feat)
            weights.append(weight)
        
        # 归一化权重
        weights = torch.cat(weights, dim=1)
        weights = F.softmax(weights, dim=1)
        
        # 加权融合特征
        fused = torch.zeros_like(features[0])
        for i, feat in enumerate(features):
            fused = fused + feat * weights[:, i:i+1]
        
        # 应用层归一化
        fused = self.layer_norm(fused)
        
        return fused
    
    def forward(self, features):
        """支持检查点的前向传播"""
        if self.use_checkpointing and self.training:
            return checkpoint.checkpoint(self._forward_impl, features)
        else:
            return self._forward_impl(features)


class OptimizedFeaturePyramidNetwork(nn.Module):
    """
    优化的特征金字塔网络，降低显存占用
    """
    def __init__(self, in_channels_list, out_channels):
        super(OptimizedFeaturePyramidNetwork, self).__init__()
        
        # 降低中间特征维度
        mid_channels = out_channels // 2
        
        # 侧向连接，使用1x1卷积降低通道数
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
            )
        
        # 输出卷积，使用深度可分离卷积降低参数量
        self.fpn_convs = nn.ModuleList()
        for _ in range(len(in_channels_list)):
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, 
                             groups=mid_channels, bias=False),  # 深度卷积
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)  # 逐点卷积
                )
            )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """高效初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, features):
        """优化的前向传播实现"""
        # 应用侧向卷积
        laterals = []
        for lateral_conv, feature in zip(self.lateral_convs, features):
            lateral = lateral_conv(feature)
            laterals.append(lateral)
        
        # 自顶向下的路径
        for i in range(len(laterals) - 1, 0, -1):
            # 上采样低分辨率特征
            if laterals[i].shape[2:] != laterals[i-1].shape[2:]:
                upsampled = F.interpolate(
                    laterals[i],
                    size=laterals[i-1].shape[2:],
                    mode='nearest'  # 使用最近邻插值代替双线性插值，更高效
                )
                laterals[i-1] = laterals[i-1] + upsampled
            else:
                laterals[i-1] = laterals[i-1] + laterals[i]
        
        # 应用特征整合卷积
        fpn_features = []
        for i, (lateral, fpn_conv) in enumerate(zip(laterals, self.fpn_convs)):
            # 应用卷积
            fpn_feature = fpn_conv(lateral)
            fpn_features.append(fpn_feature)
        
        return fpn_features


class OptimizedMultiScaleDetectionHead(nn.Module):
    """
    优化的多尺度菌落检测头，减少显存占用
    """
    def __init__(self, in_channels, num_classes, num_anchors=9,
                 predict_growth=True, scales=[0.25, 1.0, 4.0]):
        super(OptimizedMultiScaleDetectionHead, self).__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.predict_growth = predict_growth
        self.scales = scales
        
        # 降低中间层通道数
        mid_channels = in_channels // 2
        
        # 为每个尺度创建单独的检测头，使用轻量级设计
        self.cls_heads = nn.ModuleList()
        self.reg_heads = nn.ModuleList()
        
        for _ in scales:
            # 分类头 - 使用深度可分离卷积减少参数
            self.cls_heads.append(nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, 
                        groups=in_channels, bias=False),  # 深度卷积
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, num_anchors * num_classes, kernel_size=1, bias=False)  # 逐点卷积
            ))
            
            # 回归头 - 使用深度可分离卷积减少参数
            self.reg_heads.append(nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1,
                        groups=in_channels, bias=False),  # 深度卷积
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, num_anchors * 4, kernel_size=1, bias=False)  # 逐点卷积
            ))
        
        # 生长预测头 (如果启用)
        if predict_growth:
            self.growth_heads = nn.ModuleList()
            for _ in scales:
                self.growth_heads.append(nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1,
                            groups=in_channels, bias=False),  # 深度卷积
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_channels, num_anchors, kernel_size=1, bias=False),  # 逐点卷积
                    nn.Sigmoid()
                ))
        else:
            self.growth_heads = None
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """高效初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _adjust_scale(self, bbox_pred, scale):
        """根据尺度调整边界框预测"""
        # 提取宽高通道
        w_channels = bbox_pred[:, 2::4, :, :]  # 宽度通道
        h_channels = bbox_pred[:, 3::4, :, :]  # 高度通道
        
        # 应用缩放 (使用原地操作)
        w_channels.mul_(scale)
        h_channels.mul_(scale)
        
        return bbox_pred
    
    def forward(self, x):
        """优化的前向传播实现"""
        cls_scores = []
        bbox_preds = []
        growth_preds = [] if self.predict_growth else None
        
        for i, scale in enumerate(self.scales):
            # 应用分类头
            cls_score = self.cls_heads[i](x)
            cls_scores.append(cls_score)
            
            # 应用回归头
            bbox_pred = self.reg_heads[i](x)
            
            # 应用缩放比例调整边界框大小
            bbox_pred = self._adjust_scale(bbox_pred, scale)
            bbox_preds.append(bbox_pred)
            
            # 应用生长预测头（如果启用）
            if self.predict_growth:
                growth_pred = self.growth_heads[i](x)
                growth_preds.append(growth_pred)
        
        if self.predict_growth:
            return cls_scores, bbox_preds, growth_preds
        else:
            return cls_scores, bbox_preds


class FocalLoss(nn.Module):
    """
    Focal Loss实现，用于处理类别不平衡
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # 确保targets在与inputs相同的设备上
        targets = targets.to(inputs.device)
        
        # 处理输入形状
        if inputs.dim() == 3:
            B, A, C = inputs.shape
            inputs = inputs.reshape(-1, C)
            targets = targets.reshape(-1)
        
        # 保存非负类的掩码（用于处理-1填充值）
        valid_mask = targets >= 0
        if not valid_mask.all():
            inputs = inputs[valid_mask]
            targets = targets[valid_mask]
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def bbox_iou(boxes1, boxes2):
    """高效计算两组边界框之间的IoU"""
    # 确保输入是张量且在同一设备上
    if not isinstance(boxes1, torch.Tensor):
        boxes1 = torch.tensor(boxes1, dtype=torch.float32)
    if not isinstance(boxes2, torch.Tensor):
        boxes2 = torch.tensor(boxes2, dtype=torch.float32)
    
    boxes2 = boxes2.to(boxes1.device)
    
    # 获取boxes1的面积 [N]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    
    # 获取boxes2的面积 [M]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # 计算交集
    # 使用广播计算左上角和右下角
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    # 计算宽高，并确保非负
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    
    # 计算交集面积
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    # 计算并集面积
    union = area1[:, None] + area2 - inter
    
    # 计算IoU
    iou = inter / (union + 1e-8)
    
    return iou


def nms(boxes, scores, iou_threshold=0.5):
    """优化的非极大值抑制实现"""
    # 如果没有边界框，直接返回空列表
    if boxes.shape[0] == 0:
        return []
    
    # 确保输入是张量且在同一设备上
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes, dtype=torch.float32)
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float32)
    
    scores = scores.to(boxes.device)
    
    # 根据分数降序排序
    _, order = scores.sort(descending=True)
    
    keep = []
    while order.numel() > 0:
        # 保留分数最高的一个
        i = order[0].item()
        keep.append(i)
        
        # 如果只剩下一个边界框，结束循环
        if order.numel() == 1:
            break
        
        # 计算剩余边界框与当前最高分数边界框的IoU
        current_box = boxes[i:i+1]
        other_boxes = boxes[order[1:]]
        ious = bbox_iou(current_box, other_boxes)
        
        # 保留IoU小于阈值的框
        mask = ious[0] <= iou_threshold
        order = order[1:][mask]
    
    return keep


class OptimizedDetectionResultsMerger(nn.Module):
    """
    优化的检测结果合并器，显著降低显存占用
    """
    def __init__(self, image_size, iou_threshold=0.5, score_threshold=0.05):
        super(OptimizedDetectionResultsMerger, self).__init__()
        self.image_size = image_size
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
    
    def forward(self, tile_results, tile_positions):
        """优化的结果合并实现"""
        # 确保输入结果在相同设备上
        if not tile_results:
            return torch.zeros((0, 7), device='cpu')
            
        main_device = tile_results[0].device
        
        all_boxes = []
        all_scores = []
        all_labels = []
        all_growths = []
        
        # 逐块处理结果
        for result, pos in zip(tile_results, tile_positions):
            h_start, h_end, w_start, w_end = pos
            
            # 确保结果在正确的设备上
            result = result.to(main_device)
            
            # 跳过空结果
            if result.shape[0] == 0:
                continue
            
            # 提取边界框、分数、类别和生长状态
            boxes = result[:, :4]
            scores = result[:, 4]
            labels = result[:, 5].long()
            growths = result[:, 6] if result.shape[1] > 6 else torch.zeros_like(scores)
            
            # 筛选高分数检测
            mask = scores > self.score_threshold
            if not mask.any():
                continue
            
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            growths = growths[mask]
            
            # 调整边界框坐标到原始图像
            boxes[:, 0] += w_start  # x1
            boxes[:, 2] += w_start  # x2
            boxes[:, 1] += h_start  # y1
            boxes[:, 3] += h_start  # y2
            
            # 限制坐标在图像范围内 (使用原地操作)
            boxes[:, 0].clamp_(min=0, max=self.image_size)
            boxes[:, 1].clamp_(min=0, max=self.image_size)
            boxes[:, 2].clamp_(min=0, max=self.image_size)
            boxes[:, 3].clamp_(min=0, max=self.image_size)
            
            # 收集结果
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_growths.append(growths)
        
        # 如果没有检测结果，返回空张量
        if not all_boxes:
            return torch.zeros((0, 7), device=main_device)
        
        # 合并所有结果
        all_boxes = torch.cat(all_boxes, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_growths = torch.cat(all_growths, dim=0)
        
        # 分批处理NMS以减少显存使用
        batch_size = 5000  # 每批处理的检测数量
        all_keep_inds = []
        
        # 对每个类别分别进行NMS
        for class_id in torch.unique(all_labels):
            class_mask = (all_labels == class_id)
            if not class_mask.any():
                continue
            
            class_boxes = all_boxes[class_mask]
            class_scores = all_scores[class_mask]
            class_inds = torch.nonzero(class_mask, as_tuple=True)[0]
            
            # 如果数量很大，分批处理
            if len(class_boxes) > batch_size:
                # 分批处理NMS
                for i in range(0, len(class_boxes), batch_size):
                    end = min(i + batch_size, len(class_boxes))
                    batch_boxes = class_boxes[i:end]
                    batch_scores = class_scores[i:end]
                    batch_inds = class_inds[i:end]
                    
                    # 应用NMS
                    nms_keep = nms(batch_boxes, batch_scores, self.iou_threshold)
                    
                    # 收集全局索引
                    all_keep_inds.extend(batch_inds[nms_keep].tolist())
            else:
                # 直接处理
                nms_keep = nms(class_boxes, class_scores, self.iou_threshold)
                all_keep_inds.extend(class_inds[nms_keep].tolist())
        
        # 如果没有目标被保留，返回空结果
        if not all_keep_inds:
            return torch.zeros((0, 7), device=main_device)
        
        # 整合最终结果
        result_bboxes = all_boxes[all_keep_inds]
        result_scores = all_scores[all_keep_inds].unsqueeze(1)
        result_labels = all_labels[all_keep_inds].unsqueeze(1).float()
        result_growths = all_growths[all_keep_inds].unsqueeze(1)
        
        # 合并结果 [x1, y1, x2, y2, score, class_id, is_growing]
        result = torch.cat([result_bboxes, result_scores, result_labels, result_growths], dim=1).contiguous()
        
        return result


class VeritasOD(nn.Module):
    """
    优化的VeritasOD目标检测模型，为大图像(4000x4000)设计
    
    主要优化：
    1. 流式处理和分块策略保留
    2. 使用多尺度特征提取和处理
    3. 显式内存管理关键点优化
    4. 移除双向CFC，只使用标准CFC
    """
    def __init__(
        self,
        num_classes,
        feature_dim=64,              # 降低至64，原为128
        sequence_length=40,
        hidden_size_cfc_path1=32,
        hidden_size_cfc_path2=32,
        sparsity_level=0.5,
        cfc_seed=22222,
        output_size_cfc_path1=8,
        output_size_cfc_path2=8,
        data_mode='normal',
        language='en',
        image_size=4000,
        use_checkpoint=False,
        use_bidirectional_cfc=False,  # 默认为False，忽略此参数
        num_anchors=9,
        tile_size=512,                # 降低至512，原为1024
        overlap_ratio=0.25,
        fusion_units=32,
        fusion_output_size=30
    ):
        super(VeritasOD, self).__init__()
        # 核心参数
        self.sequence_length = sequence_length
        self.data_mode = data_mode
        self.language = language
        self.feature_dim = feature_dim
        self.use_checkpoint = use_checkpoint
        self.image_size = image_size
        self.use_bidirectional_cfc = False  # 强制设为False
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        
        # 存储所有构造函数参数作为属性
        self.hidden_size_cfc_path1 = hidden_size_cfc_path1
        self.hidden_size_cfc_path2 = hidden_size_cfc_path2
        self.fusion_units = fusion_units
        self.fusion_output_size = fusion_output_size
        self.sparsity_level = sparsity_level
        self.cfc_seed = cfc_seed
        self.output_size_cfc_path1 = output_size_cfc_path1
        self.output_size_cfc_path2 = output_size_cfc_path2
        
        # 基础特征提取器 - 为大图像优化
        self.base_extractor = LargeImageCNNFeatureExtractor(
            feature_dim=feature_dim,
            num_conv_layers=3,        # 减少层数，原为5
            base_channels=32,         # 减少通道数，原为64
            image_size=image_size,
            use_checkpointing=use_checkpoint,
            return_multi_scale=True
        )
        
        # 流式分块处理器 - 为大图像保留
        self.tiled_processor = StreamTiledImageProcessor(
            base_processor=self.base_extractor,
            tile_size=tile_size,
            overlap_ratio=overlap_ratio,
            max_batch_size=1,          # 限制每批最多处理1个图块，降低峰值显存
            dynamic_tile_size=True
        )
        
        # 特征金字塔网络
        self.fpn = OptimizedFeaturePyramidNetwork(
            in_channels_list=[feature_dim] * 3,  # 假设有3个特征层级
            out_channels=feature_dim
        )
        
        # ----- 路径1: 时空特征提取
        self.path1_veritp = OptimizedVeritpModule(
            in_dim=feature_dim,
            use_checkpointing=use_checkpoint
        )
        
        # 仅使用标准CFC
        self.path1_cfc = CfCWrapper(
            input_size=feature_dim,
            units=AutoNCP(
                units=hidden_size_cfc_path1,
                output_size=output_size_cfc_path1,
                sparsity_level=sparsity_level,
                seed=cfc_seed
            ),
            activation=lecun_tanh
        )
        self.output_layer_cfc_path1 = nn.Linear(output_size_cfc_path1, feature_dim, bias=False)
        
        # ----- 路径2: 卷积增强时序特征
        self.path2_veritp = OptimizedVeritpModule(
            in_dim=feature_dim,
            use_checkpointing=use_checkpoint
        )
        
        # 仅使用标准CFC
        self.path2_cfc = CfCWrapper(
            input_size=feature_dim,
            units=AutoNCP(
                units=hidden_size_cfc_path2,
                output_size=output_size_cfc_path2,
                sparsity_level=sparsity_level,
                seed=cfc_seed
            ),
            activation=lecun_tanh
        )
        self.output_layer_cfc_path2 = nn.Linear(output_size_cfc_path2, feature_dim, bias=False)
        
        # 增强模式下的注意力双模态融合
        if self.data_mode == 'enhanced':
            self.dual_modality_fusion = EfficientAttentionFusion(
                feature_dim=feature_dim,
                use_checkpointing=use_checkpoint
            )
        
        # 菌落生长特征提取器
        self.growth_feature_extractor = OptimizedColonyGrowthExtractor(
            feature_dim=feature_dim,
            use_checkpointing=use_checkpoint
        )
        
        # 路径融合 - 保留原有融合模块
        self.fusion = EfficientPathFusion(
            feature_dim=feature_dim,
            num_paths=2,
            language=language,
            use_checkpointing=use_checkpoint
        )
        
        # 多尺度菌落检测头 - 减少尺度数量
        self.detection_heads = nn.ModuleList()
        for _ in range(3):  # 为三个FPN层级创建检测头
            self.detection_heads.append(
                OptimizedMultiScaleDetectionHead(
                    in_channels=feature_dim,
                    num_classes=num_classes,
                    num_anchors=num_anchors,
                    predict_growth=True,
                    scales=[0.25, 1.0, 4.0]  # 减少到3个尺度
                )
            )
        
        # 结果合并模块
        self.results_merger = OptimizedDetectionResultsMerger(
            image_size=image_size,
            iou_threshold=0.5,
            score_threshold=0.05
        )
        
        # 定义锚框生成器
        base_anchors = self._generate_base_anchors()
        self.register_buffer('base_anchors', torch.tensor(base_anchors, dtype=torch.float32))
        
        # 内置损失函数
        self.criterion_cls = FocalLoss(alpha=1, gamma=2, reduction='mean')
        self.criterion_reg = nn.SmoothL1Loss(reduction='mean')
        
        # 初始化权重
        self._initialize_weights()
    
    def _generate_base_anchors(self):
        """生成基础锚框"""
        # 减少锚框数量，专注于最可能的菌落尺寸
        base_anchors = [
            # 小菌落锚框
            [8, 8], [6, 10], [10, 6],
            # 中等菌落锚框
            [32, 32], [26, 38], [38, 26],
            # 大菌落锚框
            [128, 128], [104, 152], [152, 104]
        ]
        return base_anchors
    
    def _initialize_weights(self):
        """高效初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def enable_gradient_checkpointing(self):
        """启用梯度检查点以减少内存使用"""
        self.use_checkpoint = True
        self.base_extractor.use_checkpointing = True
    
    def disable_gradient_checkpointing(self):
        """禁用梯度检查点"""
        self.use_checkpoint = False
        self.base_extractor.use_checkpointing = False
    
    # 安全获取data_mode属性的方法
    def get_data_mode(self):
        """安全获取数据模式，兼容DataParallel"""
        return self.data_mode
    
    def _extract_single_frame_features(self, x):
        """优化的单帧特征提取"""
        # 使用流式分块处理器提取特征
        results = self.tiled_processor(x)
        
        features = []
        positions = []
        
        for feat, pos in results:
            features.append(feat)
            positions.append(pos)
        
        return features, positions
    

    def _process_time_sequence(self, features_seq):
        """
        优化的时序处理函数
        
        Args:
            features_seq: 时序特征序列 [B, T, C]
            
        Returns:
            处理后的时序特征 [B, C]
        """
        # 路径1: 注意力增强时序特征
        path1_features = self.path1_veritp(features_seq)
        
        # 为CfC调整维度顺序
        path1_features_t = path1_features.permute(1, 0, 2).contiguous()  # [T, B, C]
        
        # 使用标准CFC处理时序依赖
        path1_cfc_out, _ = self.path1_cfc(path1_features_t)
        path1_cfc_out = path1_cfc_out.permute(1, 0, 2).contiguous()  # [B, T, C]
        
        # 输出层
        path1_output = self.output_layer_cfc_path1(path1_cfc_out)
        path1_temporal = path1_output.mean(dim=1)  # [B, C]
        
        # 路径2: 卷积增强时序特征
        path2_features = self.path2_veritp(features_seq)
        
        # 为CfC调整维度顺序
        path2_features_t = path2_features.permute(1, 0, 2).contiguous()  # [T, B, C]
        
        # 使用标准CFC处理时序依赖
        path2_cfc_out, _ = self.path2_cfc(path2_features_t)
        path2_cfc_out = path2_cfc_out.permute(1, 0, 2).contiguous()  # [B, T, C]
        
        # 输出层
        path2_output = self.output_layer_cfc_path2(path2_cfc_out)
        path2_temporal = path2_output.mean(dim=1)  # [B, C]
        
        # 融合两条路径的结果
        path_features = [path1_temporal, path2_temporal]
        fused_features = self.fusion(path_features)
        
        return fused_features
    
    def _process_frames_in_chunks(self, x):
        """
        优化的帧序列处理，适用于大图像处理 VeritasOD
        
        Args:
            x: 输入视频序列 [B,T,C,H,W]
            
        Returns:
            处理后的特征序列 [B,T,feature_dim]
        """
        batch_size, seq_len, c, h, w = x.shape
        features_seq = []
        
        for t in range(seq_len):
            # 逐帧处理
            frame = x[:, t]
            
            # 使用流式分块处理器提取特征
            results = self.tiled_processor(frame)
            
            # 合并所有图块的特征
            frame_features = []
            for feat, _ in results:
                # 处理多尺度特征情况
                if isinstance(feat, tuple):
                    frame_features.append(feat[0])  # 只取主干特征
                else:
                    frame_features.append(feat)
            
            # 计算平均特征
            if frame_features:
                frame_feature = torch.mean(torch.stack(frame_features), dim=0)
                features_seq.append(frame_feature)
        
        # 堆叠成序列 [B,T,feature_dim]
        features_seq = torch.stack(features_seq, dim=1).contiguous()
        
        return features_seq
        
    def forward(self, inputs):
        """
        高效的前向传播函数，流式处理视频序列和大图像
        
        Args:
            inputs: 对于normal模式，形状为[B,T,C,H,W]；
                  对于enhanced模式，为一个元组((x1,x2))
                  
        Returns:
            检测结果列表，每个元素为 [N, 7] 张量，包含 [x1, y1, x2, y2, score, class_id, is_growing]
        """
        # 处理不同模式的输入
        if self.get_data_mode() == 'enhanced':
            if isinstance(inputs, tuple) and len(inputs) == 2:
                x1, x2 = inputs  # tuple, x1是背光数据，x2是正常光照数据
            else:
                # 处理可能的错误输入格式
                raise ValueError(f"增强模式需要元组输入 (x1, x2)，但收到: {type(inputs)}")
        else:
            if isinstance(inputs, tuple) and len(inputs) > 0:
                x1 = inputs[0]  # 如果是元组，取第一个元素
            else:
                x1 = inputs  # 普通模式只使用背光数据
            x2 = None
        
        # 分块处理背光序列
        features1_seq = self._process_frames_in_chunks(x1)
        
        # 处理正常光照数据（如果是增强模式）
        if self.get_data_mode() == 'enhanced' and x2 is not None:
            features2_seq = self._process_frames_in_chunks(x2)
            
            # 注意力双模态融合
            features = self.dual_modality_fusion(features1_seq, features2_seq)
        else:
            # 普通模式只使用背光数据
            features = features1_seq
        
        # 提取菌落生长特征
        growth_features = self.growth_feature_extractor(features)
        
        # 将生长特征与原始特征融合 (跳过第一帧以对齐)
        features_aligned = features[:, 1:]
        features_enhanced = features_aligned + 0.2 * growth_features  # 加权融合
        
        # 时序特征处理
        temporal_features = self._process_time_sequence(features_enhanced)
        
        # 用于最终检测的帧 - 使用最后一帧
        last_frame = x1[:, -1]
        
        # 提取最后一帧的多尺度特征
        last_features, positions = self._extract_single_frame_features(last_frame)
        
        # 保存检测结果
        all_results = []
        
        # 分块处理每个区域的检测
        for i, (feat, pos) in enumerate(zip(last_features, positions)):
            if isinstance(feat, tuple):
                # 使用多尺度特征进行检测
                feat_main, multi_scale = feat
                
                # 应用FPN
                fpn_features = self.fpn(multi_scale)
                
                # 获取特征图尺寸
                feature_sizes = [f.shape[2:] for f in fpn_features]
                
                # 生成锚框
                anchors = self.generate_anchors(feature_sizes)
                
                # 应用检测头
                all_cls_scores = []
                all_bbox_preds = []
                all_growth_preds = []
                
                # 逐层处理特征
                for level, (fpn_feat, head) in enumerate(zip(fpn_features, self.detection_heads)):
                    # 融入时序特征 - 将时序特征调整为特征图形状
                    b, c = fpn_feat.shape[:2]
                    h, w = fpn_feat.shape[2:]
                    
                    # 将时序特征转换为特征图大小
                    temp_feat = temporal_features.view(b, c, 1, 1).expand(b, c, h, w)
                    
                    # 融合特征
                    enhanced_feat = fpn_feat + 0.2 * temp_feat
                    
                    # 应用检测头
                    cls_scores, bbox_preds, growth_preds = head(enhanced_feat)
                    
                    all_cls_scores.extend(cls_scores)
                    all_bbox_preds.extend(bbox_preds)
                    all_growth_preds.extend(growth_preds)
                
                # 对每个图块进行后处理
                tile_results = self.post_process(
                    cls_scores=all_cls_scores,
                    bbox_preds=all_bbox_preds,
                    growth_preds=all_growth_preds,
                    anchors=anchors
                )
                
                all_results.extend(tile_results)
            else:
                # 单尺度特征处理，跳过
                continue
        
        # 合并所有图块的检测结果
        final_results = self.results_merger(all_results, positions)
        
        return final_results
    
    def generate_anchors(self, feature_sizes):
        """
        生成每个特征图的锚框
        """
        # 获取基础锚框
        base_anchors = self.base_anchors
        
        anchors = []
        for feature_size in feature_sizes:
            # 生成网格点
            h, w = feature_size
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=base_anchors.device),
                torch.arange(w, device=base_anchors.device),
                indexing='ij'
            )
            
            # 展平网格
            grid_x = grid_x.reshape(-1)
            grid_y = grid_y.reshape(-1)
            
            # 计算中心点
            cx = (grid_x + 0.5) * 16  # 假设特征图步长为16
            cy = (grid_y + 0.5) * 16
            
            # 重复中心点以匹配不同形状的锚框
            cx = cx.repeat(self.num_anchors)
            cy = cy.repeat(self.num_anchors)
            
            # 扩展锚框宽高
            w_arr = base_anchors[:, 0].repeat(h * w)
            h_arr = base_anchors[:, 1].repeat(h * w)
            
            # 组装锚框 [x1, y1, x2, y2]
            anchor = torch.stack(
                [cx - w_arr / 2, cy - h_arr / 2, cx + w_arr / 2, cy + h_arr / 2],
                dim=1
            ).contiguous()
            
            anchors.append(anchor)
        
        return anchors
    
    def post_process(self, cls_scores, bbox_preds, growth_preds, anchors,
                    score_threshold=0.05, nms_threshold=0.5, growth_threshold=0.5, max_detections=200):
        """
        优化的检测结果后处理
        
        Args:
            cls_scores: 分类得分列表
            bbox_preds: 边界框预测列表
            growth_preds: 生长预测列表
            anchors: 锚框列表
            score_threshold: 分数阈值
            nms_threshold: NMS阈值
            growth_threshold: 菌落生长判定阈值
            max_detections: 最大检测数量
            
        Returns:
            检测结果列表
        """
        batch_size = cls_scores[0].size(0)
        num_levels = len(cls_scores)
        results = []
        
        # 每个批次分别处理
        for batch_idx in range(batch_size):
            # 收集所有预测结果
            level_bboxes = []
            level_scores = []
            level_labels = []
            level_growths = []
            
            # 逐层处理预测结果
            for level in range(num_levels):
                # 获取当前层级的结果
                cls_score = cls_scores[level][batch_idx]
                bbox_pred = bbox_preds[level][batch_idx]
                growth_pred = growth_preds[level][batch_idx]
                anchor = anchors[level]
                
                # 重塑分类得分 [A*num_classes, H, W] -> [H*W*A, num_classes]
                h, w = cls_score.size(1), cls_score.size(2)
                cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.num_classes).contiguous()
                
                # 重塑边界框预测 [A*4, H, W] -> [H*W*A, 4]
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4).contiguous()
                
                # 重塑生长预测 [A, H, W] -> [H*W*A]
                growth_pred = growth_pred.permute(1, 2, 0).reshape(-1).contiguous()
                
                # 应用分数阈值
                scores, labels = cls_score.max(dim=1)
                mask = scores > score_threshold
                
                if mask.sum() == 0:
                    continue
                
                # 筛选超过阈值的预测
                scores = scores[mask]
                labels = labels[mask]
                bboxes = bbox_pred[mask]
                anchors_filtered = anchor[mask]
                growths = growth_pred[mask]
                
                # 将边界框偏移量应用到锚框上
                widths = anchors_filtered[:, 2] - anchors_filtered[:, 0]
                heights = anchors_filtered[:, 3] - anchors_filtered[:, 1]
                ctr_x = anchors_filtered[:, 0] + 0.5 * widths
                ctr_y = anchors_filtered[:, 1] + 0.5 * heights
                
                # 应用回归偏移 (用原地操作)
                dx = bboxes[:, 0].clamp(min=-0.5, max=0.5)
                dy = bboxes[:, 1].clamp(min=-0.5, max=0.5)
                dw = bboxes[:, 2].clamp(min=-1.0, max=1.0)
                dh = bboxes[:, 3].clamp(min=-1.0, max=1.0)
                
                # 计算预测的中心点和宽高
                pred_ctr_x = dx * widths + ctr_x
                pred_ctr_y = dy * heights + ctr_y
                pred_w = torch.exp(dw) * widths
                pred_h = torch.exp(dh) * heights
                
                # 转换为 [x1, y1, x2, y2] 格式
                pred_x1 = pred_ctr_x - 0.5 * pred_w
                pred_y1 = pred_ctr_y - 0.5 * pred_h
                pred_x2 = pred_ctr_x + 0.5 * pred_w
                pred_y2 = pred_ctr_y + 0.5 * pred_h
                
                # 限制边界框在图像范围内 (使用原地操作)
                pred_x1.clamp_(min=0, max=self.image_size)
                pred_y1.clamp_(min=0, max=self.image_size)
                pred_x2.clamp_(min=0, max=self.image_size)
                pred_y2.clamp_(min=0, max=self.image_size)
                pred_bboxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1).contiguous()
                
                # 收集结果
                level_bboxes.append(pred_bboxes)
                level_scores.append(scores)
                level_labels.append(labels)
                level_growths.append(growths)
            
            # 合并所有层级的结果
            if len(level_bboxes) == 0:
                # 如果没有检测到任何目标，返回空结果
                results.append(torch.zeros((0, 7), device=cls_scores[0].device))
                continue
            
            # 合并所有层级结果
            bboxes = torch.cat(level_bboxes, dim=0)
            scores = torch.cat(level_scores, dim=0)
            labels = torch.cat(level_labels, dim=0)
            growths = torch.cat(level_growths, dim=0)
            
            # 按类别分批处理NMS，降低显存使用
            keep_inds = []
            unique_labels = torch.unique(labels)
            
            # 逐类处理NMS
            for class_id in unique_labels:
                class_mask = (labels == class_id)
                if not class_mask.any():
                    continue
                
                class_boxes = bboxes[class_mask]
                class_scores = scores[class_mask]
                class_inds = torch.nonzero(class_mask, as_tuple=True)[0]
                
                # 分批处理大量检测框
                if len(class_boxes) > 1000:
                    # 分批次处理
                    batch_size = 1000
                    
                    for i in range(0, len(class_boxes), batch_size):
                        end_idx = min(i + batch_size, len(class_boxes))
                        batch_boxes = class_boxes[i:end_idx]
                        batch_scores = class_scores[i:end_idx]
                        batch_inds = class_inds[i:end_idx]
                        
                        # 应用NMS
                        batch_keep = nms(batch_boxes, batch_scores, nms_threshold)
                        keep_inds.extend(batch_inds[batch_keep].tolist())
                else:
                    # 直接处理
                    nms_keep = nms(class_boxes, class_scores, nms_threshold)
                    keep_inds.extend(class_inds[nms_keep].tolist())
            
            # 如果没有目标被保留，返回空结果
            if len(keep_inds) == 0:
                results.append(torch.zeros((0, 7), device=cls_scores[0].device))
                continue
            
            # 限制检测数量
            keep_inds = keep_inds[:max_detections]
            
            # 获取保留的结果
            result_bboxes = bboxes[keep_inds]
            result_scores = scores[keep_inds].unsqueeze(1)
            result_labels = labels[keep_inds].unsqueeze(1).float()
            
            # 判断是否为生长中的菌落
            result_growths = (growths[keep_inds] > growth_threshold).float().unsqueeze(1)
            
            # 组合结果 [x1, y1, x2, y2, score, class_id, is_growing]
            result = torch.cat([result_bboxes, result_scores, result_labels, result_growths], dim=1).contiguous()
            results.append(result)
        
        return results
    
    def compute_loss(self, outputs, targets, loss_type='auto', class_counts=None, language='en'):
        """计算检测损失函数"""
        # 确保输入在同一设备上
        if isinstance(targets, torch.Tensor):
            targets = targets.to(outputs.device)
        elif isinstance(targets, tuple):
            targets = tuple(t.to(outputs.device) if isinstance(t, torch.Tensor) else t for t in targets)

        if isinstance(outputs, tuple) and len(outputs) >= 2:
            # 如果outputs是元组（如分类得分和回归预测）
            cls_logits, bbox_preds = outputs[:2]
            
            if isinstance(targets, tuple) and len(targets) == 3:
                # Detection targets (bboxes, labels, seq_id)
                gt_bboxes, gt_labels, _ = targets

                if gt_bboxes.shape[0] == 0:
                    # 无目标 => 视为背景 (class 0)
                    gt_label = torch.tensor(0, dtype=torch.long, device=cls_logits.device)
                    gt_bbox = torch.zeros(4, dtype=torch.float32, device=cls_logits.device)
                else:
                    gt_label = gt_labels[0]
                    gt_bbox = gt_bboxes[0]

                # 分类损失
                loss_cls = nn.CrossEntropyLoss()(cls_logits, gt_label.unsqueeze(0))

                # 回归损失 - 使用GIoU损失代替平滑L1损失，提高检测性能
                def bbox_to_xyxy(bbox):
                    """将[x,y,w,h]转换为[x1,y1,x2,y2]"""
                    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                    return torch.tensor([x, y, x+w, y+h], device=bbox.device)

                def calculate_giou(pred_bbox, gt_bbox):
                    """计算GIoU损失"""
                    pred_xyxy = bbox_to_xyxy(pred_bbox)
                    gt_xyxy = bbox_to_xyxy(gt_bbox)

                    # 计算交集区域
                    x1 = torch.max(pred_xyxy[0], gt_xyxy[0])
                    y1 = torch.max(pred_xyxy[1], gt_xyxy[1])
                    x2 = torch.min(pred_xyxy[2], gt_xyxy[2])
                    y2 = torch.min(pred_xyxy[3], gt_xyxy[3])

                    # 交集面积
                    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

                    # 预测框面积
                    pred_area = (pred_xyxy[2] - pred_xyxy[0]) * (pred_xyxy[3] - pred_xyxy[1])

                    # 真实框面积
                    gt_area = (gt_xyxy[2] - gt_xyxy[0]) * (gt_xyxy[3] - gt_xyxy[1])

                    # 并集面积
                    union = pred_area + gt_area - intersection

                    # IoU
                    iou = intersection / (union + 1e-7)

                    # 最小外接矩形的坐标
                    c_x1 = torch.min(pred_xyxy[0], gt_xyxy[0])
                    c_y1 = torch.min(pred_xyxy[1], gt_xyxy[1])
                    c_x2 = torch.max(pred_xyxy[2], gt_xyxy[2])
                    c_y2 = torch.max(pred_xyxy[3], gt_xyxy[3])

                    # 最小外接矩形的面积
                    c_area = (c_x2 - c_x1) * (c_y2 - c_y1)

                    # GIoU
                    giou = iou - (c_area - union) / (c_area + 1e-7)

                    # GIoU损失
                    giou_loss = 1 - giou

                    return giou_loss

                loss_reg = calculate_giou(bbox_preds[0], gt_bbox)

                # 自动选择合适的损失函数
                if loss_type == 'auto' and class_counts:
                    max_c = max(class_counts.values())
                    min_c = min(class_counts.values())
                    if max_c / (min_c + 1e-6) > 3:
                        # 类别不平衡严重时使用focal loss
                        loss_cls = FocalLoss(alpha=1, gamma=2, reduction='mean')(cls_logits, gt_label.unsqueeze(0))

                # 总损失 - 使用更合理的权重，提高目标定位精度
                total_loss = loss_cls + 0.5 * loss_reg  # 增加回归损失的权重
                return total_loss, loss_cls, loss_reg
        
        # 如果不是检测任务输出格式，返回零损失
        return torch.tensor(0.0, device=outputs.device, requires_grad=True), \
               torch.tensor(0.0, device=outputs.device, requires_grad=True), \
               torch.tensor(0.0, device=outputs.device, requires_grad=True)
