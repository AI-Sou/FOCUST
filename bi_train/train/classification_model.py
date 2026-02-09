# train/classification_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms.functional import rgb_to_grayscale
import os
import math # 引入math模块

# 使用相对导入
try:
    from ..ncps.wrapper import CfCWrapper
    from ..ncps.wirings import AutoNCP
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        from bi_train.ncps.wrapper import CfCWrapper
        from bi_train.ncps.wirings import AutoNCP
    except ImportError:
        # 最后尝试直接导入
        from ncps.wrapper import CfCWrapper
        from ncps.wirings import AutoNCP

# --- 辅助模块定义 ---

class DepthwiseSeparableConv(nn.Module):
    """
    高效的深度可分离卷积块。这是模型效率的基础。
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x); x = self.bn1(x); x = self.silu(x)
        x = self.pointwise(x); x = self.bn2(x); x = self.silu(x)
        return x

class InvertedResidualBlock(nn.Module):
    """
    高效的倒置残差块（MBConv）。
    """
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride
        hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.SiLU(inplace=True))
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        ])
        
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class BioGrowthNetV2(nn.Module):
    """
    为生物生长序列设计的BioGrowthNetV2。
    """
    def __init__(
        self, feature_dim=512, dropout_rate=0.4,
        initial_channels=32, stage_channels=[48, 96, 192],
        num_blocks=[2, 3, 2], expand_ratios=[4, 4, 4]
    ):
        super(BioGrowthNetV2, self).__init__()
        self.backbone = nn.ModuleList()
        self.backbone.append(DepthwiseSeparableConv(3, initial_channels, kernel_size=3, stride=2, padding=1))
        
        in_channels = initial_channels
        for i in range(len(stage_channels)):
            out_ch = stage_channels[i]
            n_blocks = num_blocks[i]
            exp_ratio = expand_ratios[i]
            self.backbone.append(InvertedResidualBlock(in_channels, out_ch, stride=2, expand_ratio=exp_ratio))
            for _ in range(n_blocks - 1):
                self.backbone.append(InvertedResidualBlock(out_ch, out_ch, stride=1, expand_ratio=exp_ratio))
            in_channels = out_ch
            
        self.backbone = nn.Sequential(*self.backbone)
        
        self.final_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_channels, feature_dim)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        return self.final_head(features)

class FocalLoss(nn.Module):
    """Focal Loss 实现，用于处理类别不平衡。"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
    def forward(self, inputs, targets):
        targets = targets.to(inputs.device); ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss); focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum() if self.reduction == 'sum' else focal_loss

class TemporalAttention(nn.Module):
    def __init__(self, feature_dim):
        super(TemporalAttention, self).__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )

    def forward(self, x):
        attention_scores = self.attention_net(x)
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_features = x * attention_weights
        context_vector = torch.sum(weighted_features, dim=1)
        return context_vector

class CrossAttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super(CrossAttentionFusion, self).__init__()
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.scale = math.sqrt(feature_dim)
        self.output_layer = nn.Linear(feature_dim, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, query_source, context_source):
        query = self.query_proj(query_source.unsqueeze(1))
        key = self.key_proj(context_source.unsqueeze(1))
        value = self.value_proj(context_source.unsqueeze(1))
        attention_scores = torch.bmm(query, key.transpose(1, 2)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_context = torch.bmm(attention_weights, value).squeeze(1)
        fused_output = self.output_layer(attended_context)
        final_output = self.layer_norm(query_source + fused_output)
        return final_output

class Focust(nn.Module):
    """
    【最终修复版】Focust模型。
    """
    def __init__(
        self, num_classes, feature_dim, hidden_size_cfc, output_size_cfc,
        fusion_hidden_size, sparsity_level=0.5, cfc_seed=22222, dropout_rate=0.5,
        batch_first=False,
        **kwargs
    ):
        super(Focust, self).__init__()
        self.feature_dim = feature_dim
        self.batch_first = batch_first
        
        arc_kwargs = {
            'initial_channels': kwargs.get('initial_channels', 32),
            'stage_channels': kwargs.get('stage_channels', [48, 96, 192]),
            'num_blocks': kwargs.get('num_blocks', [2, 3, 2]),
            'expand_ratios': kwargs.get('expand_ratios', [4, 4, 4])
        }
        
        self.feature_extractor = BioGrowthNetV2(
            feature_dim=feature_dim,
            dropout_rate=dropout_rate,
            **arc_kwargs
        )
        
        self.path1_weight = nn.Linear(feature_dim, feature_dim)
        self.temporal_attention = TemporalAttention(feature_dim)
        
        self.cfc_path1 = CfCWrapper(
            input_size=feature_dim,
            units=AutoNCP(units=hidden_size_cfc, output_size=output_size_cfc, sparsity_level=sparsity_level, seed=cfc_seed),
            return_sequences=False, batch_first=self.batch_first)
        self.cfc_path2 = CfCWrapper(
            input_size=feature_dim,
            units=AutoNCP(units=hidden_size_cfc, output_size=output_size_cfc, sparsity_level=sparsity_level, seed=cfc_seed + 1),
            return_sequences=False, batch_first=self.batch_first)
        
        self.fusion_attention = CrossAttentionFusion(output_size_cfc)
        
        self.fusion_classifier = nn.Sequential(
            nn.Linear(output_size_cfc, fusion_hidden_size), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate), nn.Linear(fusion_hidden_size, num_classes))

        self._fix_cfc_buffers()

        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_focal = FocalLoss(alpha=1, gamma=2, reduction='mean')
        self.criterion_mse = nn.MSELoss()

    def _fix_cfc_buffers(self):
        """一次性检查并修复CfC模块以确保多GPU兼容性。"""
        for path_name in ['cfc_path1', 'cfc_path2']:
            cfc_module = getattr(self, path_name)
            if hasattr(cfc_module, 'units') and hasattr(cfc_module.units, 'sparsity_mask'):
                if 'sparsity_mask' not in cfc_module.units._buffers:
                    mask_tensor = cfc_module.units.sparsity_mask
                    delattr(cfc_module.units, 'sparsity_mask')
                    cfc_module.units.register_buffer('sparsity_mask', mask_tensor)

    def forward(self, inputs):
        """前向传播，包含对cuDNN错误的最终修复。"""
        if isinstance(inputs, (list, tuple)):
            inputs = inputs[0]
            
        batch_size, seq_len, c, h, w = inputs.size()
        
        # --- 核心修复：解决CUDNN_STATUS_NOT_SUPPORTED错误 ---
        # view 操作会创建非连续的Tensor视图，而cuDNN的某些内核要求输入是连续的。
        # 调用 .contiguous() 会创建一个新的、内存连续的Tensor副本，从而解决此问题。
        x_flat = inputs.view(batch_size * seq_len, c, h, w).contiguous()
        
        features_flat = self.feature_extractor(x_flat)
        features = features_flat.view(batch_size, seq_len, self.feature_dim).contiguous()

        # --- 后续操作同样保持良好的内存连续性习惯，增加代码的鲁棒性 ---
        last_frame_features = features[:, -1, :].contiguous()
        weighted_last = self.path1_weight(last_frame_features).unsqueeze(1)
        weighted_seq = features * weighted_last.expand_as(features)
        path1_input_features = weighted_seq
        
        global_features = self.temporal_attention(features).contiguous()
        # expand_as 也可能产生非连续视图，增加 contiguous() 调用
        path2_input_features = global_features.unsqueeze(1).expand_as(features).contiguous()
        
        if not self.batch_first:
            # permute 必定产生非连续视图，必须调用 contiguous()
            path1_input_features = path1_input_features.permute(1, 0, 2).contiguous()
            path2_input_features = path2_input_features.permute(1, 0, 2).contiguous()

        path1_output, _ = self.cfc_path1(path1_input_features)
        path2_output, _ = self.cfc_path2(path2_input_features)
        
        fused_features = self.fusion_attention(path1_output, path2_output)
        
        return self.fusion_classifier(fused_features)

    def compute_loss(self, outputs, targets, loss_type='auto', class_counts=None, language='en'):
        """根据配置自动选择或指定损失函数。"""
        targets = targets.to(outputs.device)
        if loss_type == 'auto':
            if class_counts and min(class_counts.values()) > 0 and (max(class_counts.values()) / min(class_counts.values()) > 2.0):
                return self.criterion_focal(outputs, targets)
            return self.criterion_ce(outputs, targets)
        elif loss_type == 'cross_entropy': 
            return self.criterion_ce(outputs, targets)
        elif loss_type == 'focal': 
            return self.criterion_focal(outputs, targets)
        elif loss_type == 'mse': 
            return self.criterion_mse(outputs, F.one_hot(targets, num_classes=outputs.shape[1]).float())
        else: 
            raise ValueError(f"未知的损失函数类型: {loss_type}")
            
    def get_data_mode(self):
        return "normal"