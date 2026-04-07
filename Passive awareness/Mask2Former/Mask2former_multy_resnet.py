import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models import build_segmentor
from mmengine.config import ConfigDict

# 保持 MultiScaleRegressionHead 不变，逻辑是通用的
class MultiScaleRegressionHead(nn.Module):
    def __init__(self, in_channels_list, out_channels, hidden_dim=256):
        """
        in_channels_list: Backbone输出的各层通道数
        out_channels: 输出通道数 (T=1, v=2)
        hidden_dim: 融合后的中间层通道数
        """
        super().__init__()
        
        self.linear_layers = nn.ModuleList()
        for in_ch in in_channels_list:
            self.linear_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, hidden_dim, kernel_size=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU()
                )
            )
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.pred_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, inputs):
        target_h, target_w = inputs[0].shape[2], inputs[0].shape[3]
        outs = []
        for i, x in enumerate(inputs):
            x = self.linear_layers[i](x)
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
            outs.append(x)
        out = torch.cat(outs, dim=1) 
        out = self.fusion_conv(out)
        out = self.pred_conv(out)
        return out

class TeXMask2Former_Resnet(nn.Module):
    def __init__(self, nclass=30, in_channels=49, train_T=True, train_v=True):
        super().__init__()
        self.nclass = nclass
        self.train_T = train_T
        self.train_v = train_v

        # ------------------------------------------------------------------
        # [关键修改 1] 定义 ResNet-50 的输出通道列表
        # ResNet50 的 stage 1-4 输出通道默认为 [256, 512, 1024, 2048]
        # ------------------------------------------------------------------
        resnet_out_channels = [256, 512, 1024, 2048]

        cfg_dict = ConfigDict(
            type='EncoderDecoder',
            # ------------------------------------------------------------------
            # [关键修改 2] Backbone 设置为 ResNet-50 并修改输入通道
            # ------------------------------------------------------------------
            backbone=dict(
                type='ResNet',
                depth=50,
                in_channels=in_channels, # 这里设置为 49
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                dilations=(1, 1, 1, 1),
                strides=(1, 2, 2, 2),
                norm_cfg=dict(type='BN', requires_grad=True),
                norm_eval=False,
                style='pytorch',
                contract_dilation=True,
                # 可以加载预训练权重，第一层形状不匹配时会自动警告并重新初始化，其他层正常加载
                init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
            ),
            decode_head=dict(
                type='Mask2FormerHead',
                # [关键修改 3] 必须匹配 ResNet-50 的输出通道
                in_channels=resnet_out_channels, 
                strides=[4, 8, 16, 32],
                feat_channels=256,
                out_channels=256,
                num_classes=nclass,
                num_queries=100,
                num_transformer_feat_level=3,
                align_corners=False,
                
                # Pixel Decoder 配置 (保持你原有的配置)
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
                    positional_encoding=dict(
                        num_feats=128, 
                        normalize=True
                    ),
                    init_cfg=None),
                
                enforce_decoder_input_project=False,
                
                positional_encoding=dict(
                    num_feats=128, 
                    normalize=True
                ),
                
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
        
        # 构建模型
        self.mmseg_model = build_segmentor(cfg_dict)
        
        # ------------------------------------------------------------------
        # [关键修改 4] 使用 ResNet 通道数初始化回归头
        # ------------------------------------------------------------------
        if self.train_T:
            self.T_head = MultiScaleRegressionHead(
                in_channels_list=resnet_out_channels, # [256, 512, 1024, 2048]
                out_channels=1, 
                hidden_dim=256
            )
            
        if self.train_v:
            self.v_head = MultiScaleRegressionHead(
                in_channels_list=resnet_out_channels, 
                out_channels=2, 
                hidden_dim=256
            )

    def forward(self, x):
        # 1. 获取 Backbone 特征
        feat = self.mmseg_model.extract_feat(x)
        # feat 应该是 [B, 256, H/4, W/4], [B, 512, H/8, W/8], ...
        
        # 2. 预测 eMap (材质语义分割)
        batch_size = x.shape[0]
        batch_img_metas = [{
            'img_shape': x.shape[-2:], 
            'ori_shape': x.shape[-2:],
            'pad_shape': x.shape[-2:], 
            'scale_factor': 1.0
        }] * batch_size

        e_logits = self.mmseg_model.encode_decode(x, batch_img_metas=batch_img_metas)
        
        # 3. 预测 T 和 v
        T_pred = None
        v_pred = None
        
        # 注意：这里我们传入整个 feat 列表给 RegressionHead 做多尺度融合
        if self.train_T:
            T_pred = self.T_head(feat)
            if T_pred.shape[-2:] != x.shape[-2:]:
                T_pred = F.interpolate(T_pred, size=x.shape[-2:], mode='bilinear', align_corners=False)
                
        if self.train_v:
            v_pred = self.v_head(feat)
            if v_pred.shape[-2:] != x.shape[-2:]:
                v_pred = F.interpolate(v_pred, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # 4. 拼装输出
        output = e_logits
        
        if self.train_T:
            output = torch.cat([output, T_pred], dim=1)
        if self.train_v:
            output = torch.cat([output, v_pred], dim=1)
            
        return output