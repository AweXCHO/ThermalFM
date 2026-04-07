import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from timm_swin_mae import SwinMAE, SwinMAE_ablation
from models_infmae_skip4 import MaskedAutoencoderInfMAE
from models_mae import MaskedAutoencoderViT
import torchvision.models as models

import timm

class TCIC_convnext(nn.Module):
    def __init__(self, num_classes=3, model_name="convnext_base"):
        super().__init__()
        
        # 使用timm创建预训练模型
        self.encoder = timm.create_model(model_name, pretrained=True)
        
        # 获取特征维度
        feature_dim = self.encoder.num_features
        
        # 移除分类头
        self.backbone = self.encoder.forward_features
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
class TCIC_resnet50(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        # 加载预训练的ResNet50
        self.encoder = models.resnet50(pretrained=pretrained)
        
        # 移除ResNet50最后的全连接层和平均池化层
        # 保留直到最后一个卷积层之前的所有层
        self.backbone = nn.Sequential(*list(self.encoder.children())[:-2])
        
        # ResNet50最后一个卷积层的输出通道数是2048
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(2048, num_classes)  # 修改为2048
        
    def forward(self, x):
        # 通过ResNet50 backbone
        x = self.backbone(x)  # 输出形状: (B, 2048, 7, 7)
        
        x = self.avgpool(x)   # 输出形状: (B, 2048, 1, 1)
        x = torch.flatten(x, 1)  # 输出形状: (B, 2048)
        x = self.dropout(x)
        x = self.fc(x)        # 输出形状: (B, num_classes)
        return x

class TCIC_swinmae(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.encoder = SwinMAE()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = x[-1]
        # (B, 1024, 7, 7)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class TCIC_swinmae_ablation(nn.Module):
    def __init__(self, num_classes=3, atm=True, pretrain_pth = ''):
        super().__init__()
        self.encoder = SwinMAE_ablation(atm=atm, pretrain_pth=pretrain_pth)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = x[-1]
        # (B, 1024, 7, 7)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
class TCIC_infmae(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.encoder = MaskedAutoencoderInfMAE(
                img_size=[224, 56, 28], patch_size=[4, 2, 2], embed_dim=[256, 384, 768], depth=[2, 2, 11], num_heads=12,
                decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
                mlp_ratio=[4, 4, 4])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(49 * 768, num_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    

class TCIC_mae(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.encoder = self.encoder = MaskedAutoencoderViT(
            patch_size=16, embed_dim=1024, depth=24, num_heads=16,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(50 * 1024, num_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x