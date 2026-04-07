import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models import build_segmentor
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from .timm_swin_mae import SwinMAE

# ==============================================================================
# 1. SegFormerRegressionHead (推荐方案)
#    结构简单高效：MLP 投影 -> 上采样拼接 -> 融合 -> 预测
#    比 Transformer Decoder 更容易训练，对回归任务极其有效
# ==============================================================================
class SegFormerRegressionHead(BaseModule):
    def __init__(self, in_channels_list, embedding_dim=256, dropout_ratio=0.1):
        """
        Args:
            in_channels_list: Backbone 输出通道 [256, 512, 1024, 1024]
            embedding_dim: 统一映射后的通道数
        """
        super().__init__()
        self.in_channels_list = in_channels_list
        self.num_stages = len(in_channels_list)
        
        # 1. 线性投影层：将不同层的通道数统一映射到 embedding_dim
        self.linear_layers = nn.ModuleList()
        for in_channels in in_channels_list:
            self.linear_layers.append(
                nn.Conv2d(in_channels, embedding_dim, kernel_size=1)
            )
        
        # 2. 融合层：将拼接后的特征 (embedding_dim * 4) 进行融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(embedding_dim * self.num_stages, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_ratio)
        )

        # 3. T 预测头 (Transmittance)
        self.T_pred = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(),
            nn.Conv2d(embedding_dim, 1, kernel_size=1),
            nn.Sigmoid() # 物理约束 [0, 1]
        )
        
        # 4. v 预测头 (Scattering)
        self.v_pred = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(),
            nn.Conv2d(embedding_dim, 2, kernel_size=1),
            nn.Softplus() # 物理约束 > 0
        )

    def forward(self, inputs):
        # inputs: [c1, c2, c3, c4]
        
        outs = []
        # 目标尺寸：取第一层特征的尺寸 (H/4, W/4)
        target_h, target_w = inputs[0].shape[2], inputs[0].shape[3]

        for i, x in enumerate(inputs):
            # 1. MLP 统一通道
            x = self.linear_layers[i](x)
            
            # 2. 上采样到统一尺寸 (H/4, W/4)
            if i > 0:
                x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
            
            outs.append(x)

        # 3. 拼接 (Concat)
        out = torch.cat(outs, dim=1) # [B, 256*4, H/4, W/4]
        
        # 4. 融合 (Fusion)
        out = self.fusion_conv(out)  # [B, 256, H/4, W/4]
        
        # 5. 预测
        t_out = self.T_pred(out)
        v_out = self.v_pred(out)
        
        return t_out, v_out

# ==============================================================================
# 2. TeXMask2Former 主模型
# ==============================================================================
class TeXMask2SegFormer(nn.Module):
    def __init__(self, nclass, in_channels=49, train_T=True, train_v=True):
        super().__init__()
        self.nclass = 30
        self.train_T = train_T
        self.train_v = train_v

        # SwinMAE 输出通道
        swin_mae_out_channels = [256, 512, 1024, 1024]

        # ------------------------------------------------------------------
        # 1. eMap 分支 (保持 Mask2Former 不变，因为它做分类很好)
        # ------------------------------------------------------------------
        cfg_dict = ConfigDict(
            type='EncoderDecoder',
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
                in_channels=swin_mae_out_channels, 
                strides=[4, 8, 16, 32],
                feat_channels=256,
                out_channels=256,
                num_classes=nclass,
                num_queries=100,
                num_transformer_feat_level=3,
                align_corners=False,
                pixel_decoder=dict(
                    type='mmdet.MSDeformAttnPixelDecoder',
                    num_outs=3,
                    norm_cfg=dict(type='GN', num_groups=32),
                    act_cfg=dict(type='ReLU'),
                    encoder=dict(num_layers=6, layer_cfg=dict(self_attn_cfg=dict(embed_dims=256, num_heads=8, num_levels=3, num_points=4, im2col_step=64, dropout=0.0, batch_first=True, norm_cfg=None, init_cfg=None), ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, num_fcs=2, ffn_drop=0.0, act_cfg=dict(type='ReLU', inplace=True))), init_cfg=None),
                    positional_encoding=dict(num_feats=128, normalize=True), init_cfg=None),
                enforce_decoder_input_project=False,
                positional_encoding=dict(num_feats=128, normalize=True),
                transformer_decoder=dict(return_intermediate=True, num_layers=9, layer_cfg=dict(self_attn_cfg=dict(embed_dims=256, num_heads=8, attn_drop=0.0, proj_drop=0.0, dropout_layer=None, batch_first=True), cross_attn_cfg=dict(embed_dims=256, num_heads=8, attn_drop=0.0, proj_drop=0.0, dropout_layer=None, batch_first=True), ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, num_fcs=2, act_cfg=dict(type='ReLU', inplace=True), ffn_drop=0.0, dropout_layer=None, add_identity=True)), init_cfg=None),
                loss_cls=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0, reduction='mean', class_weight=[1.0] * nclass + [0.1]),
                loss_mask=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=5.0),
                loss_dice=dict(type='mmdet.DiceLoss', use_sigmoid=True, activate=True, reduction='mean', naive_dice=True, eps=1.0, loss_weight=5.0)
            ),
            train_cfg=dict(num_points=12544, oversample_ratio=3.0, importance_sample_ratio=0.75, assigner=dict(type='mmdet.HungarianAssigner', match_costs=[dict(type='mmdet.ClassificationCost', weight=2.0), dict(type='mmdet.CrossEntropyLossCost', weight=5.0, use_sigmoid=True), dict(type='mmdet.DiceCost', weight=5.0, pred_act=True, eps=1.0)]), sampler=dict(type='mmdet.MaskPseudoSampler')),
            test_cfg=dict(mode='slide', crop_size=(224, 224), stride=(56, 56))
        )
        
        # 1. 构建 eMap 分割模型
        self.mmseg_model = build_segmentor(cfg_dict)
        
        # 2. 替换 Backbone 为 SwinMAE
        self.my_backbone = SwinMAE(in_channels=in_channels)
        self.mmseg_model.backbone = self.my_backbone
        
        # ------------------------------------------------------------------
        # 3. [关键] 使用 SegFormer 风格的回归头
        # ------------------------------------------------------------------
        if self.train_T or self.train_v:
            self.tv_head = SegFormerRegressionHead(
                in_channels_list=swin_mae_out_channels,
                embedding_dim=256
            )

    def forward(self, x):
        # 1. 提取特征
        feat = self.mmseg_model.extract_feat(x)
        
        # 2. 预测 eMap
        batch_size = x.shape[0]
        batch_img_metas = [{'img_shape': x.shape[-2:], 'ori_shape': x.shape[-2:], 'pad_shape': x.shape[-2:], 'scale_factor': 1.0}] * batch_size
        e_logits = self.mmseg_model.encode_decode(x, batch_img_metas=batch_img_metas)
        
        # 3. 预测 T 和 v
        T_pred = None
        v_pred = None
        
        if hasattr(self, 'tv_head'):
            # SegFormer Head 直接输出 H/4, W/4 的结果
            t_out, v_out = self.tv_head(feat)
            
            # 双线性上采样回原图
            if self.train_T:
                T_pred = F.interpolate(t_out, size=x.shape[-2:], mode='bilinear', align_corners=False)
            if self.train_v:
                v_pred = F.interpolate(v_out, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # 4. 拼接
        output = e_logits
        if self.train_T:
            output = torch.cat([output, T_pred], dim=1)
        if self.train_v:
            output = torch.cat([output, v_pred], dim=1)
            
        return output