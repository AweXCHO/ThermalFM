import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from timm_swin_mae import SwinMAE, SwinMAE_ablation
import torchvision.models as models

import timm

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