import torch
import torch.nn as nn
import torch.nn.functional as F

# 引入构建工具
from mmseg.models import build_segmentor
from mmengine.config import ConfigDict
from mmengine.registry import MODELS 
from mmengine.model import BaseModule

# 假设 timm_swin_mae 在同一目录下，保持引用不变
from .timm_swin_mae import SwinMAE

# ==============================================================================
# 1. Mask2FormerRegressionHead
#    专门用于 T 和 v 的回归头。它复用了 Mask2Former 中最强大的 
#    Pixel Decoder (MSDeformAttn) 来提取高质量特征。
# ==============================================================================
class Mask2FormerRegressionHead(BaseModule):
    def __init__(self, in_channels_list, hidden_dim=256):
        """
        Args:
            in_channels_list: Backbone输出的通道列表 [256, 512, 1024, 1024]
            hidden_dim: Pixel Decoder 内部维度
        """
        super().__init__()
        
        # 定义 Pixel Decoder 配置
        # 这是 Mask2Former 的核心：多尺度可变形注意力 Transformer
        pixel_decoder_cfg = ConfigDict(
            type='mmdet.MSDeformAttnPixelDecoder',
            in_channels=in_channels_list,    # 自动适配 SwinMAE 的输出
            strides=[4, 8, 16, 32],          # 对应特征图的步长
            feat_channels=hidden_dim,
            out_channels=hidden_dim,
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(  # Transformer Encoder 配置
                num_layers=6,
                layer_cfg=dict(  
                    self_attn_cfg=dict(  
                        embed_dims=hidden_dim,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfg=dict(
                        embed_dims=hidden_dim,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True))),
                init_cfg=None),
            positional_encoding=dict(
                num_feats=128, 
                normalize=True
            ),
            init_cfg=None
        )
        
        # 构建 Pixel Decoder
        # 注意：这里会调用 mmdet 的注册表，确保环境安装了 mmdet
        self.pixel_decoder = MODELS.build(pixel_decoder_cfg)
        
        # 定义预测头 (Convolutional Heads)
        # Pixel Decoder 的输出特征 mask_feature 分辨率是 H/4, W/4
        
        # T (Transmittance): 范围 [0, 1] -> Sigmoid
        self.T_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid() 
        )
        
        # v (Volumetric Scattering): 范围 > 0 -> Softplus
        self.v_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 2, 1), # 输出2通道
            nn.Softplus()
        )

    def forward(self, features):
        # features 是 Backbone 输出的列表
        
        # 1. 经过 Pixel Decoder 融合特征
        # 返回值: (mask_features, multi_scale_features)
        # mask_features 是分辨率最高 (Stride 4) 且经过 Transformer 融合的特征
        mask_features, _ = self.pixel_decoder(features)
        
        # 2. 回归预测
        T_out = self.T_head(mask_features)
        v_out = self.v_head(mask_features)
        
        return T_out, v_out

# ==============================================================================
# 2. TeXMask2Former 主模型
# ==============================================================================
class TeXMask2Former(nn.Module):
    def __init__(self, nclass, in_channels=49, train_T=True, train_v=True):
        super().__init__()
        self.nclass = 30
        self.train_T = train_T
        self.train_v = train_v

        # 你的 SwinMAE 输出通道配置
        swin_mae_out_channels = [256, 512, 1024, 1024]

        # ------------------------------------------------------------------
        # 1. 构建基础 Mask2Former (用于 eMap 分类)
        # ------------------------------------------------------------------
        cfg_dict = ConfigDict(
            type='EncoderDecoder',
            # 这里的 backbone 只是占位，后面会被 self.my_backbone 替换
            backbone=dict(
                type='ResNet', 
                depth=50,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                norm_cfg=dict(type='BN', requires_grad=True),
                norm_eval=False,
                style='pytorch',
                contract_dilation=True
            ),
            decode_head=dict(
                type='Mask2FormerHead',
                in_channels=swin_mae_out_channels, # 必须匹配 SwinMAE
                strides=[4, 8, 16, 32],
                feat_channels=256,
                out_channels=256,
                num_classes=nclass,
                num_queries=100,
                num_transformer_feat_level=3,
                align_corners=False,
                
                # --- eMap 专用的 Pixel Decoder ---
                pixel_decoder=dict(
                    type='mmdet.MSDeformAttnPixelDecoder',
                    num_outs=3,
                    norm_cfg=dict(type='GN', num_groups=32),
                    act_cfg=dict(type='ReLU'),
                    encoder=dict(  
                        num_layers=6,
                        layer_cfg=dict(  
                            self_attn_cfg=dict(  
                                embed_dims=256,
                                num_heads=8,
                                num_levels=3,
                                num_points=4,
                                im2col_step=64,
                                dropout=0.0,
                                batch_first=True,
                                norm_cfg=None,
                                init_cfg=None),
                            ffn_cfg=dict(
                                embed_dims=256,
                                feedforward_channels=1024,
                                num_fcs=2,
                                ffn_drop=0.0,
                                act_cfg=dict(type='ReLU', inplace=True))),
                        init_cfg=None),
                    positional_encoding=dict(num_feats=128, normalize=True),
                    init_cfg=None),
                
                enforce_decoder_input_project=False,
                positional_encoding=dict(num_feats=128, normalize=True),
                
                # --- eMap 专用的 Transformer Decoder ---
                transformer_decoder=dict(
                    return_intermediate=True,
                    num_layers=9,
                    layer_cfg=dict(  
                        self_attn_cfg=dict(  
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.0,
                            proj_drop=0.0,
                            dropout_layer=None,
                            batch_first=True),
                        cross_attn_cfg=dict(  
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.0,
                            proj_drop=0.0,
                            dropout_layer=None,
                            batch_first=True),
                        ffn_cfg=dict(
                            embed_dims=256,
                            feedforward_channels=2048,
                            num_fcs=2,
                            act_cfg=dict(type='ReLU', inplace=True),
                            ffn_drop=0.0,
                            dropout_layer=None,
                            add_identity=True)),
                    init_cfg=None),
                
                # Loss 配置
                loss_cls=dict(
                    type='mmdet.CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=2.0,
                    reduction='mean',
                    class_weight=[1.0] * nclass + [0.1]),
                loss_mask=dict(
                    type='mmdet.CrossEntropyLoss',
                    use_sigmoid=True,
                    reduction='mean',
                    loss_weight=5.0),
                loss_dice=dict(
                    type='mmdet.DiceLoss',
                    use_sigmoid=True,
                    activate=True,
                    reduction='mean',
                    naive_dice=True,
                    eps=1.0,
                    loss_weight=5.0)
            ),
            train_cfg=dict(
                num_points=12544,
                oversample_ratio=3.0,
                importance_sample_ratio=0.75,
                assigner=dict(
                    type='mmdet.HungarianAssigner',
                    match_costs=[
                        dict(type='mmdet.ClassificationCost', weight=2.0),
                        dict(type='mmdet.CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                        dict(type='mmdet.DiceCost', weight=5.0, pred_act=True, eps=1.0)
                    ]),
                sampler=dict(type='mmdet.MaskPseudoSampler')),
            test_cfg=dict(mode='slide', crop_size=(224, 224), stride=(56, 56))
        )
        
        # 1. 构建模型
        self.mmseg_model = build_segmentor(cfg_dict)
        
        # 2. [核心替换] 替换 Backbone 为 SwinMAE
        self.my_backbone = SwinMAE(in_channels=in_channels)
        self.mmseg_model.backbone = self.my_backbone
        
        # ------------------------------------------------------------------
        # 3. 构建 T 和 v 的 Mask2Former 回归头
        # ------------------------------------------------------------------
        if self.train_T or self.train_v:
            # 实例化我们定义的 Mask2FormerRegressionHead
            self.tv_head = Mask2FormerRegressionHead(
                in_channels_list=swin_mae_out_channels,
                hidden_dim=256
            )

    def forward(self, x):
        # 1. 特征提取 (Backbone)
        # feat 结构: [B, 256, H/4, W/4], [B, 512, H/8, W/8], ...
        feat = self.mmseg_model.extract_feat(x)
        
        # 2. 预测 eMap (走原始 Mask2Former 路径)
        batch_size = x.shape[0]
        batch_img_metas = [{
            'img_shape': x.shape[-2:], 
            'ori_shape': x.shape[-2:],
            'pad_shape': x.shape[-2:], 
            'scale_factor': 1.0
        }] * batch_size

        e_logits = self.mmseg_model.encode_decode(x, batch_img_metas=batch_img_metas)
        
        # 3. 预测 T 和 v (走新的 Mask2Former Pixel Decoder 路径)
        T_pred = None
        v_pred = None
        
        if hasattr(self, 'tv_head'):
            # 获取特征
            t_out, v_out = self.tv_head(feat)
            
            # Pixel Decoder 输出是 H/4, W/4，需要上采样回原图大小
            if self.train_T:
                T_pred = F.interpolate(t_out, size=x.shape[-2:], mode='bilinear', align_corners=False)
                
            if self.train_v:
                v_pred = F.interpolate(v_out, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # 4. 拼接输出
        # 输出顺序: [e_logits, T, v]
        output = e_logits
        
        if self.train_T:
            output = torch.cat([output, T_pred], dim=1)
        if self.train_v:
            output = torch.cat([output, v_pred], dim=1)
            
        return output