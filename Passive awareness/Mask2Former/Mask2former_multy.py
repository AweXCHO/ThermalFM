import torch
import torch.nn as nn
import torch.nn.functional as F
# 引入 mmseg 的构建工具
from mmseg.models import build_segmentor
from .timm_swin_mae import SwinMAE
from mmengine.config import ConfigDict

class MultiScaleRegressionHead(nn.Module):
    def __init__(self, in_channels_list, out_channels, hidden_dim=256):
        """
        in_channels_list: Backbone输出的各层通道数，例如 [256, 512, 1024, 1024]
        out_channels: 输出通道数 (T=1, v=2)
        hidden_dim: 融合后的中间层通道数
        """
        super().__init__()
        
        # 1. MLP/Conv 层：将不同通道数的特征统一映射到 hidden_dim
        self.linear_layers = nn.ModuleList()
        for in_ch in in_channels_list:
            self.linear_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, hidden_dim, kernel_size=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU()
                )
            )
        
        # 2. 融合层：将拼接后的特征 (hidden_dim * 4) 融合
        # 4 是因为有 4 层特征
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1) # 防止过拟合
        )
        
        # 3. 输出层：回归预测
        self.pred_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, inputs):
        # inputs 是一个列表/元组 [x0, x1, x2, x3]
        # x0: 1/4, x1: 1/8, x2: 1/16, x3: 1/32 (相对于原图)
        
        target_h, target_w = inputs[0].shape[2], inputs[0].shape[3]
        
        outs = []
        for i, x in enumerate(inputs):
            # 先调整通道数
            x = self.linear_layers[i](x)
            
            # 再上采样到统一尺寸 (x0 的尺寸)
            x = F.interpolate(x, size=(target_h * 2, target_w * 2), mode='bilinear', align_corners=False)
            
            outs.append(x)
            
        # 拼接
        out = torch.cat(outs, dim=1) # [B, hidden_dim*4, H/4, W/4]
        
        # 融合
        out = self.fusion_conv(out)
        
        # 预测
        out = self.pred_conv(out)
        
        return out
    
class TeXMask2Former(nn.Module):
    def __init__(self, nclass, in_channels=49, train_T=True, train_v=True):
        super().__init__()
        self.nclass = 30
        self.train_T = train_T
        self.train_v = train_v

        my_backbone_out_channels = [256, 512, 1024, 1024]

        # ------------------------------------------------------------------
        # 1. 定义 Mask2Former 配置 (这里以 Swin-Tiny 为例，可换 ResNet)
        # ------------------------------------------------------------------
        # 你可以从 mmseg 的 config 文件里复制这部分，或者加载文件
        cfg_dict = ConfigDict(
            type='EncoderDecoder',
            # 占位符 Backbone
            backbone=dict(
                type='ResNet', 
                depth=18,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                norm_cfg=dict(type='BN', requires_grad=True),
                norm_eval=False,
                style='pytorch',
                contract_dilation=True
            ),
            decode_head=dict(
                type='Mask2FormerHead',
                in_channels=[256, 512, 1024, 1024], # 必须匹配 SwinMAE 输出
                strides=[4, 8, 16, 32],
                feat_channels=256,
                out_channels=256,
                num_classes=nclass,
                num_queries=100,
                num_transformer_feat_level=3,
                align_corners=False,
                
                # -----------------------------------------------------------
                # Pixel Decoder (需保留 type，因为是被 build 调用的)
                # -----------------------------------------------------------
                pixel_decoder=dict(
                    type='mmdet.MSDeformAttnPixelDecoder',
                    num_outs=3,
                    norm_cfg=dict(type='GN', num_groups=32),
                    act_cfg=dict(type='ReLU'),
                    
                    # [内部组件 1] encoder: 直接实例化，无 type
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
                    
                    # [内部组件 2] PE: 直接实例化，无 type
                    positional_encoding=dict(
                        num_feats=128, 
                        normalize=True
                    ),
                    init_cfg=None),
                
                enforce_decoder_input_project=False,
                
                # -----------------------------------------------------------
                # [关键修正] 外层 PE: MMDet 3.x 也是直接实例化，必须去掉 type
                # -----------------------------------------------------------
                positional_encoding=dict(
                    # type='SinePositionalEncoding', <-- 删除这行
                    num_feats=128, 
                    normalize=True
                ),
                
                # -----------------------------------------------------------
                # Transformer Decoder: 直接实例化，无 type
                # -----------------------------------------------------------
                transformer_decoder=dict(
                    # type='DetrTransformerDecoder', <-- 删除这行
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
                
                # Loss 不需要改，它们是通过 build 构建的，需要 type
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
        # Mask2Former 的骨干通常是 3 通道，我们稍后修改
        
        # 构建模型
        self.mmseg_model = build_segmentor(cfg_dict)
        
        # ------------------------------------------------------------------
        # 2. 修改输入层：3通道 -> 49通道
        # ------------------------------------------------------------------
        # 假设 backbone 是 Swin Transformer
        self.my_backbone = SwinMAE(in_channels=49)
        self.mmseg_model.backbone = self.my_backbone
        # ------------------------------------------------------------------
        # 3. 添加 T 和 v 的预测头 (Auxiliary Heads)
        # ------------------------------------------------------------------
        # Mask2Former 擅长分类 (eMap)，但不擅长回归 (T/v)。
        # 我们利用 Backbone 输出的特征图来做 T 和 v 的回归。
        # Swin-Tiny 输出特征通道通常是 [96, 192, 384, 768] (具体看 config)
        
        feat_dim = my_backbone_out_channels[0] 
        
        if self.train_T:
            # T 需要高分辨率细节，hidden_dim 可以设大一点比如 256
            self.T_head = MultiScaleRegressionHead(
                in_channels_list=my_backbone_out_channels, 
                out_channels=1, 
                hidden_dim=256
            )
            
        if self.train_v:
            # v 是平滑的，可以使用相同的结构，或者加深感受野
            self.v_head = MultiScaleRegressionHead(
                in_channels_list=my_backbone_out_channels, 
                out_channels=2, # v有2个通道
                hidden_dim=256
            )

    def forward(self, x):
        # 1. 获取 Backbone 特征
        # extract_feat 通常返回一个列表/元组 (feat1, feat2, feat3, feat4)
        feat = self.mmseg_model.extract_feat(x)
        
        # 2. 预测 eMap (材质语义分割)
        # 使用 mmseg 的 encode_decode 直接得到 logits
        # 注意：encode_decode 内部通常包含 resize 到原图大小
        batch_size = x.shape[0]
        batch_img_metas = [{
            'img_shape': x.shape[-2:], 
            'ori_shape': x.shape[-2:],
            'pad_shape': x.shape[-2:], # 最好加上 pad_shape，MMSeg 内部常会用到
            'scale_factor': 1.0
        }] * batch_size

        e_logits = self.mmseg_model.encode_decode(x, batch_img_metas=batch_img_metas)
        
        # 3. 预测 T 和 v (回归任务)
        # 这里的 feat[0] 是分辨率最高的特征 (通常是原图的 1/4)
        # 如果需要更好的效果，可以用 FPN 融合后的特征
        # Mask2Former 内部有 Pixel Decoder，如果不方便提取，直接用 Backbone 的特征最简单
        
        T_pred = None
        v_pred = None
        
        if self.train_T:
            T_pred = self.T_head(feat)
            # 确保尺寸匹配
            if T_pred.shape[-2:] != x.shape[-2:]:
                T_pred = F.interpolate(T_pred, size=x.shape[-2:], mode='bilinear', align_corners=False)
                
        if self.train_v:
            v_pred = self.v_head(feat)
            if v_pred.shape[-2:] != x.shape[-2:]:
                v_pred = F.interpolate(v_pred, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # 4. 拼装输出
        # 原代码 model.py 的 forward 输出是一个拼接的 Tensor: [e_logits, T_pred, v_pred]
        output = e_logits
        
        if self.train_T:
            output = torch.cat([output, T_pred], dim=1)
        if self.train_v:
            output = torch.cat([output, v_pred], dim=1)
            
        return output