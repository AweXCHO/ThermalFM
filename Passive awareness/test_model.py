import segmentation_models_pytorch as smp
import custom_encoder
import torch
import torch.nn as nn
model = smp.FPN(encoder_name='SwinMAE', encoder_weights=None, in_channels=49, classes=1)
# model.segmentation_head[1] = nn.Upsample(size=(448,448), mode='bilinear', align_corners=False)
y = model(torch.randn(2,49,224,224))
print(y.shape)  # -> (2,1,448,448)