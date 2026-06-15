import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from .layers import *
from .timm_swin_mae import SwinMAE

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc=[256, 512, 1024, 1024], scales=[0], num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.scales = scales

        self.num_ch_enc = num_ch_enc  # 应该传入 [256, 512, 1024, 1024]
        self.num_ch_dec = np.array([256, 512, 512, 1024, 1024])  # 调整解码器通道数

        self.convs = OrderedDict()

        # 修改循环范围为3到0，共4个层级
        for i in range(3, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 3 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}
        up_features = []
        for feat in input_features:
            feat = upsample(feat, scale_factor=4)
            up_features.append(feat)

        x = up_features[-1]
        #print(x.shape)
        for i in range(3, -1, -1):  # 循环范围改为3到0
            x = self.convs[("upconv", i, 0)](x)
            # print(i, x.shape)
            if i < 3:
                x = [upsample(x)]
            else:
                x = [x]
            if self.use_skips and i > 0:
                x = x + [up_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            #print(x.shape)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
        #print(self.outputs[("disp", 0)].shape)

        return self.outputs

class mySimple(nn.Module):
    def __init__(self, min_depth = 0.1, max_depth = 100.0):
        super(mySimple, self).__init__()
        self.encoder = SwinMAE()
        self.decoder = DepthDecoder()
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, inputs):
        feats = self.encoder(inputs)
        #print(feats.shape)
        outputs = self.decoder(feats)
        pred_depth = outputs[("disp", 0)] * self.max_depth
        #print(pred_depth)
        return pred_depth
