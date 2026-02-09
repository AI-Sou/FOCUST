import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import torch.utils.checkpoint as checkpoint
import logging
# 使用相对导入
try:
    from ..ncps.wrapper import CfCWrapper, lecun_tanh
    from ..ncps.wirings import AutoNCP
    from ..config_utils import get_message
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        from mutil_train.ncps.wrapper import CfCWrapper, lecun_tanh
        from mutil_train.ncps.wirings import AutoNCP
        from mutil_train.train.config_utils import get_message
    except ImportError:
        # 最后尝试直接导入
        from ncps.wrapper import CfCWrapper, lecun_tanh
        from ncps.wirings import AutoNCP
        # 配置工具导入失败时的处理
        def get_message(*args, **kwargs):
            return "配置工具不可用"

class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积模块，用于减少模型参数。
    相比普通卷积，参数量和计算量显著减少。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SimpleCNNFeatureExtractor(nn.Module):
    """
    简化的卷积特征提取器，使用深度可分离卷积。
    最终输出维度 = feature_dim。
    """
    def __init__(self, feature_dim=128, num_conv_layers=4, base_channels=64, image_size=224):
        super(SimpleCNNFeatureExtractor, self).__init__()
        layers = []
        in_channels = 3
        out_channels = base_channels

        for _ in range(num_conv_layers):
            layers.append(
                DepthwiseSeparableConv(
                    in_channels, out_channels,
                    kernel_size=3, stride=1, padding=1
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
            out_channels *= 2

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(in_channels, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class FocalLoss(nn.Module):
    """
    Focal Loss 实现，用于应对类别不平衡。
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 确保targets在与inputs相同的设备上
        targets = targets.to(inputs.device)
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EnhancedAttentionFusion(nn.Module):
    """
    增强的注意力融合模块，使用CfC网络生成注意力权重
    """
    def __init__(
        self,
        feature_dim,
        fusion_units,
        fusion_output_size,
        sparsity_level=0.5,
        cfc_seed=22222,
        language='en'
    ):
        super(EnhancedAttentionFusion, self).__init__()
        if fusion_output_size >= fusion_units - 2:
            raise ValueError(
                get_message(language, "fusion_output_size_exceed").format(fusion_output_size, fusion_units - 2)
            )

        self.feature_dim = feature_dim
        self.fusion_units = fusion_units
        self.fusion_output_size = fusion_output_size
        self.sparsity_level = sparsity_level
        self.cfc_seed = cfc_seed
        self.language = language

        self.ncp_attention = CfCWrapper(
            input_size=feature_dim,
            units=AutoNCP(
                units=fusion_units,
                output_size=fusion_output_size,
                sparsity_level=sparsity_level,
                seed=cfc_seed
            ),
            activation=lecun_tanh,
            return_sequences=False
        )

        self.fc = nn.Sequential(
            nn.Linear(fusion_output_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, feature_dim),
            nn.Sigmoid()
        )
        
        self.enhanced_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, features, enhanced_features=None):
        concatenated = torch.stack(features, dim=1)
        avg_features = concatenated.mean(dim=1)

        attention_output, _ = self.ncp_attention(avg_features)
        attention_weights = self.fc(attention_output)
        fused = avg_features * attention_weights
        
        if enhanced_features is not None:
            enhanced_concatenated = torch.stack(enhanced_features, dim=1)
            enhanced_avg_features = enhanced_concatenated.mean(dim=1)
            
            combined_features = torch.cat([fused, enhanced_avg_features], dim=1)
            fusion_weights = self.enhanced_fusion(combined_features)
            
            fused = fused + enhanced_avg_features * fusion_weights
            
        return fused


class Focust(nn.Module):
    """
    【最终修复版】改进版多路径时序网络。
    - 修复了forward函数中的Tensor内存连续性问题，确保与cuDNN的兼容性，解决了推理时性能骤降的Bug。
    """
    def __init__(
        self,
        num_classes,
        feature_dim,
        sequence_length,
        hidden_size_cfc_path1,
        hidden_size_cfc_path2,
        fusion_units,
        fusion_output_size,
        sparsity_level=0.5,
        cfc_seed=22222,
        output_size_cfc_path1=8,
        output_size_cfc_path2=8,
        data_mode='normal',
        language='en',
        image_size=224
    ):
        super(Focust, self).__init__()
        
        self.sequence_length = sequence_length
        self.data_mode = data_mode
        self.language = language
        self.feature_dim = feature_dim
        self.fusion_units = fusion_units
        self.fusion_output_size = fusion_output_size
        self.hidden_size_cfc_path1 = hidden_size_cfc_path1
        self.hidden_size_cfc_path2 = hidden_size_cfc_path2
        self.output_size_cfc_path1 = output_size_cfc_path1
        self.output_size_cfc_path2 = output_size_cfc_path2
        self.sparsity_level = sparsity_level
        self.cfc_seed = cfc_seed
        self.image_size = image_size

        logging.getLogger(__name__).info(
            "初始化Focust模型: num_classes=%s, feature_dim=%s, fusion_units=%s, fusion_output_size=%s, data_mode=%s, image_size=%s",
            num_classes,
            feature_dim,
            fusion_units,
            fusion_output_size,
            data_mode,
            image_size,
        )

        if fusion_output_size >= fusion_units - 2:
            raise ValueError(
                get_message(language, "fusion_output_size_exceed").format(fusion_output_size, fusion_units - 2)
            )

        self.feature_extractor = SimpleCNNFeatureExtractor(
            feature_dim=feature_dim,
            num_conv_layers=4,
            base_channels=64,
            image_size=image_size
        )

        self.path1_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
        self.path1_fc = nn.Linear(feature_dim, feature_dim)

        self.cfc_path1 = CfCWrapper(
            input_size=feature_dim,
            units=AutoNCP(
                units=hidden_size_cfc_path1,
                output_size=output_size_cfc_path1,
                sparsity_level=sparsity_level,
                seed=cfc_seed
            ),
            activation=lecun_tanh,
            return_sequences=True
        )
        self.output_layer_cfc_path1 = nn.Linear(output_size_cfc_path1, feature_dim)

        self.path2_conv = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=3,
            padding=1
        )
        self.path2_fc = nn.Linear(feature_dim, feature_dim)

        self.cfc_path2 = CfCWrapper(
            input_size=feature_dim,
            units=AutoNCP(
                units=hidden_size_cfc_path2,
                output_size=output_size_cfc_path2,
                sparsity_level=sparsity_level,
                seed=cfc_seed # 注意：种子应该不同以创建不同的网络
            ),
            activation=lecun_tanh,
            return_sequences=True
        )
        self.output_layer_cfc_path2 = nn.Linear(output_size_cfc_path2, feature_dim)

        self.fusion = EnhancedAttentionFusion(
            feature_dim=feature_dim,
            fusion_units=fusion_units,
            fusion_output_size=fusion_output_size,
            sparsity_level=sparsity_level,
            cfc_seed=cfc_seed,
            language=self.language
        )
        
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.3)
        self.output_layer = nn.Linear(feature_dim, num_classes)

        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_focal = FocalLoss(alpha=1, gamma=2, reduction='mean')
        self.criterion_mse = nn.MSELoss()

    def get_data_mode(self):
        return self.data_mode

    def process_through_path(self, features, path_attention, path_fc, cfc_path, output_layer_cfc):
        last_frame = features[:, -1, :]
        attention_weights = path_attention(last_frame)
        path = last_frame * attention_weights
        path = path_fc(path)
        
        replicated_last_frame = path.unsqueeze(1).repeat(1, features.size(1), 1)
        path_features = features * replicated_last_frame
        
        # 确保输入CfC的张量是内存连续的
        path_features_for_cfc = path_features.transpose(0, 1).contiguous()
        cfc_output, _ = cfc_path(path_features_for_cfc)
        cfc_output = cfc_output.transpose(0, 1).contiguous()
        
        temporal = output_layer_cfc(cfc_output)
        temporal = temporal.mean(dim=1)
        
        return temporal

    def forward(self, inputs) -> torch.Tensor:
        if self.get_data_mode() == 'enhanced':
            x, x2 = inputs
        else:
            x, x2 = inputs, None

        batch_size, seq_len, C, H, W = x.size()
        
        # --- 【核心Bug修复】 ---
        # 确保在将张量送入需要内存连续输入的模块（如cuDNN加速的卷积）之前，调用.contiguous()。
        # .view() 操作可能会返回一个非连续的内存视图，这是导致推理时cuDNN出错的根源。
        x_view = x.view(batch_size * seq_len, C, H, W).contiguous()
        
        features = self.feature_extractor(x_view)
        
        # 同样，在下一个 .view() 之后也确保内存连续性
        features = features.view(batch_size, seq_len, -1).contiguous()

        # Path 1
        path1_temporal = self.process_through_path(
            features, 
            self.path1_attention, 
            self.path1_fc, 
            self.cfc_path1, 
            self.output_layer_cfc_path1
        )

        # Path 2
        # .permute() 操作几乎总是返回非连续视图，必须调用 .contiguous()
        path2_features_permuted = features.permute(0, 2, 1).contiguous()
        path2_conv = self.path2_conv(path2_features_permuted)
        path2_conv = path2_conv.permute(0, 2, 1).contiguous()
        path2_conv = self.path2_fc(path2_conv)
        
        # 确保输入CfC的张量是内存连续的
        path2_features_for_cfc = path2_conv.transpose(0, 1).contiguous()
        cfc_output_path2, _ = self.cfc_path2(path2_features_for_cfc)
        cfc_output_path2 = cfc_output_path2.transpose(0, 1).contiguous()
        
        path2_temporal = self.output_layer_cfc_path2(cfc_output_path2)
        path2_temporal = path2_temporal.mean(dim=1)

        path_features = [path1_temporal, path2_temporal]
        
        enhanced_features = None
        if self.get_data_mode() == 'enhanced' and x2 is not None:
            # 对 x2 执行与 x 完全相同的、内存安全的处理流程
            x2_view = x2.view(batch_size * seq_len, C, H, W).contiguous()
            features2 = self.feature_extractor(x2_view).view(batch_size, seq_len, -1).contiguous()
            path1_temporal_enhanced = self.process_through_path(features2, self.path1_attention, self.path1_fc, self.cfc_path1, self.output_layer_cfc_path1)
            
            features2_permuted = features2.permute(0, 2, 1).contiguous()
            path2_conv_enhanced = self.path2_conv(features2_permuted).permute(0, 2, 1).contiguous()
            path2_conv_enhanced = self.path2_fc(path2_conv_enhanced)
            path2_features_for_cfc_enhanced = path2_conv_enhanced.transpose(0, 1).contiguous()
            cfc_output_path2_enhanced, _ = self.cfc_path2(path2_features_for_cfc_enhanced)
            cfc_output_path2_enhanced = cfc_output_path2_enhanced.transpose(0, 1).contiguous()
            path2_temporal_enhanced = self.output_layer_cfc_path2(cfc_output_path2_enhanced).mean(dim=1)
            
            enhanced_features = [path1_temporal_enhanced, path2_temporal_enhanced]

        fused = self.fusion(path_features, enhanced_features)
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        output = self.output_layer(fused)
        return output

    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        loss_type='auto',
        class_counts=None,
        language='en'
    ) -> torch.Tensor:
        targets = targets.to(outputs.device)
        
        if loss_type == 'auto':
            if class_counts:
                max_count = max(class_counts.values())
                min_count = min(class_counts.values())
                if min_count > 0 and (max_count / min_count > 3):
                    return self.criterion_focal(outputs, targets)
            return self.criterion_ce(outputs, targets)
        elif loss_type == 'cross_entropy':
            return self.criterion_ce(outputs, targets)
        elif loss_type == 'focal':
            return self.criterion_focal(outputs, targets)
        elif loss_type == 'mse':
            one_hot = F.one_hot(targets, num_classes=outputs.shape[1]).float()
            return self.criterion_mse(outputs, one_hot)
        else:
            raise ValueError(get_message(language, "unknown_loss_type").format(loss_type))


# 兼容旧命名：早期版本将分类模型称为 Veritas
Veritas = Focust

__all__ = [
    "DepthwiseSeparableConv",
    "SimpleCNNFeatureExtractor",
    "FocalLoss",
    "EnhancedAttentionFusion",
    "Focust",
    "Veritas",
]
