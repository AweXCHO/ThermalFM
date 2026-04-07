import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(in_ch, out_ch, kernel, stride=1, padding='same', activation=True):
    if isinstance(kernel, int):
        k = kernel
    else:
        k = kernel[0]
    if padding == 'same':
        pad = (k - 1) // 2
    elif padding == 'valid' or padding == 0:
        pad = 0
    else:
        pad = padding
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad, bias=True)
    if activation:
        return nn.Sequential(conv, nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    else:
        return nn.Sequential(conv, )  # batchnorm + activation applied after residual add where needed


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w


class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // ratio, channels, kernel_size=1, bias=True)
        )
        # the same mlp will be used for avg and max (we'll call it twice)

    def forward(self, x):
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        max_ = torch.max(x, dim=2, keepdim=True)[0]
        max_ = torch.max(max_, dim=3, keepdim=True)[0] if max_.dim() == 4 else max_
        a = self.mlp(avg)
        b = self.mlp(max_)
        scale = torch.sigmoid(a + b)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_ = torch.max(x, dim=1, keepdim=True)[0]
        cat = torch.cat([avg, max_], dim=1)
        w = torch.sigmoid(self.conv(cat))
        return x * w


class CBAMBlock(nn.Module):
    def __init__(self, channels, ratio=8, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class InceptionResNetA(nn.Module):
    def __init__(self, in_ch, scale=0.1, attention=None, reduction_ratio=8):
        super().__init__()
        # branches
        self.branch1 = _conv(in_ch, 32, 1)
        self.branch2 = nn.Sequential(
            _conv(in_ch, 32, 1),
            _conv(32, 32, 3)
        )
        self.branch3 = nn.Sequential(
            _conv(in_ch, 32, 1),
            _conv(32, 48, 3),
            _conv(48, 64, 3)
        )
        # final conv to restore channels to in_ch
        self.conv_linear = nn.Conv2d(32 + 32 + 64, in_ch, kernel_size=1)
        self.scale = scale
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)

        if attention == 'se':
            self.at = SEBlock(in_ch, reduction=reduction_ratio)
        elif attention == 'cbam':
            self.at = CBAMBlock(in_ch, ratio=reduction_ratio)
        else:
            self.at = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        init = x
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        out = torch.cat([b1, b2, b3], dim=1)
        out = self.conv_linear(out)
        out = out * self.scale
        if self.at is not None:
            out = self.at(out)
        out = init + out
        out = self.bn(out)
        out = self.relu(out)
        return out


class InceptionResNetB(nn.Module):
    def __init__(self, in_ch, scale=0.1, attention=None, reduction_ratio=8):
        super().__init__()
        self.branch1 = _conv(in_ch, 192, 1)
        self.branch2 = nn.Sequential(
            _conv(in_ch, 128, 1),
            _conv(128, 160, (1, 7)),
            _conv(160, 192, (7, 1))
        )
        self.conv_linear = nn.Conv2d(192 + 192, in_ch, kernel_size=1)  # branch1(192) + branch2(192)
        self.scale = scale
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)

        if attention == 'se':
            self.at = SEBlock(in_ch, reduction=reduction_ratio)
        elif attention == 'cbam':
            self.at = CBAMBlock(in_ch, ratio=reduction_ratio)
        else:
            self.at = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        init = x
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        out = torch.cat([b1, b2], dim=1)
        out = self.conv_linear(out)
        out = out * self.scale
        if self.at is not None:
            out = self.at(out)
        out = init + out
        out = self.bn(out)
        out = self.relu(out)
        return out


class InceptionResNetC(nn.Module):
    def __init__(self, in_ch, scale=0.1, attention=None, reduction_ratio=8):
        super().__init__()
        self.branch1 = _conv(in_ch, 192, 1)
        self.branch2 = nn.Sequential(
            _conv(in_ch, 192, 1),
            _conv(192, 224, (1, 3)),
            _conv(224, 256, (3, 1))
        )
        self.conv_linear = nn.Conv2d(192 + 256, in_ch, kernel_size=1)
        self.scale = scale
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)

        if attention == 'se':
            self.at = SEBlock(in_ch, reduction=reduction_ratio)
        elif attention == 'cbam':
            self.at = CBAMBlock(in_ch, ratio=reduction_ratio)
        else:
            self.at = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        init = x
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        out = torch.cat([b1, b2], dim=1)
        out = self.conv_linear(out)
        out = out * self.scale
        if self.at is not None:
            out = self.at(out)
        out = init + out
        out = self.bn(out)
        out = self.relu(out)
        return out


class ReductionA(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        # according to TF code: k=256, l=256, m=384, n=384
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.branch1 = _conv(in_ch, 384, 3, stride=2, padding='valid')
        self.branch2 = nn.Sequential(
            _conv(in_ch, 256, 1),
            _conv(256, 256, 3),
            _conv(256, 384, 3, stride=2, padding='valid')
        )
        self.bn = nn.BatchNorm2d(384 * 3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b1 = self.branch_pool(x)
        b2 = self.branch1(x)
        b3 = self.branch2(x)
        out = torch.cat([b1, b2, b3], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ReductionB(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.branch1 = nn.Sequential(
            _conv(in_ch, 256, 1),
            _conv(256, 384, 3, stride=2, padding='valid')
        )
        self.branch2 = nn.Sequential(
            _conv(in_ch, 256, 1),
            _conv(256, 288, 3, stride=2, padding='valid')
        )
        self.branch3 = nn.Sequential(
            _conv(in_ch, 256, 1),
            _conv(256, 288, 3),
            _conv(288, 320, 3, stride=2, padding='valid')
        )
        self.bn = nn.BatchNorm2d(in_ch + 384 + 288 + 320)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b1 = self.branch_pool(x)
        b2 = self.branch1(x)
        b3 = self.branch2(x)
        b4 = self.branch3(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Stem(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        # follow the TF implementation in your script
        self.conv1 = _conv(in_ch, 32, 3, stride=2, padding='valid')  # valid, stride2
        self.conv2 = _conv(32, 32, 3, padding='valid')
        self.conv3 = _conv(32, 64, 3)
        # split
        self.split_conv = _conv(64, 96, 3, stride=2, padding='valid')
        self.pool = nn.MaxPool2d(3, stride=2, padding=0)
        # next
        # after concat -> 160 -> two branches to make 96 and 96 => 192
        self.split_conv2_a = nn.Sequential(
            _conv(160, 64, 1),
            _conv(64, 96, 3, padding='valid')
        )
        self.split_conv2_b = nn.Sequential(
            _conv(160, 64, 1),
            _conv(64, 64, (7, 1)),
            _conv(64, 64, (1, 7)),
            _conv(64, 96, 3, padding='valid')
        )
        # reduction
        self.split_conv3 = _conv(192, 192, 3, stride=2, padding='valid')
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=0)
        self.bn = nn.BatchNorm2d(384)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        b = self.conv3(x)  # called block_1 in tf
        x1 = self.pool(b)
        x2 = self.split_conv(b)
        x = torch.cat([x1, x2], dim=1)  # 64 + 96 = 160
        a = self.split_conv2_a(x)
        b = self.split_conv2_b(x)
        x = torch.cat([a, b], dim=1)  # 96 + 96 = 192
        x1 = self.split_conv3(x)
        x2 = self.maxpool3(x)
        x = torch.cat([x1, x2], dim=1)  # 192 + 192 = 384
        x = self.bn(x)
        x = self.relu(x)
        return x


class SEInceptionResNetV2(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, attention='cbam', reduction_ratio=8, dropout=0.2):
        """
        attention: one of {None, 'se', 'cbam'}
        input size expected: (N, in_channels, 224, 224)
        """
        super().__init__()
        self.stem = Stem(in_channels)  # outputs 384 channels

        # 5 x Inception-ResNet-A (in_ch=384)
        self.repeat_a = nn.Sequential(*[InceptionResNetA(384, attention=attention, reduction_ratio=reduction_ratio) for _ in range(5)])

        # Reduction A -> outputs 1152
        self.reduction_a = ReductionA(384)

        # 10 x Inception-ResNet-B (in_ch=1152)
        self.repeat_b = nn.Sequential(*[InceptionResNetB(1152, attention=attention, reduction_ratio=reduction_ratio) for _ in range(10)])

        # Reduction B -> outputs 2144
        self.reduction_b = ReductionB(1152)

        # 5 x Inception-ResNet-C (in_ch=2144)
        self.repeat_c = nn.Sequential(*[InceptionResNetC(2144, attention=attention, reduction_ratio=reduction_ratio) for _ in range(5)])

        # final
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2144, num_classes)

        self._init_head()

    def _init_head(self):
        nn.init.kaiming_normal_(self.classifier.weight, nonlinearity='relu')
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, x):
        # x: [N, C, 224, 224]
        x = self.stem(x)                 # -> [N, 384, H, W]
        x = self.repeat_a(x)            # -> [N, 384, ...]
        x = self.reduction_a(x)         # -> [N, 1152, ...]
        x = self.repeat_b(x)            # -> [N, 1152, ...]
        x = self.reduction_b(x)         # -> [N, 2144, ...]
        x = self.repeat_c(x)            # -> [N, 2144, ...]
        x = self.avgpool(x)             # -> [N, 2144, 1, 1]
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # quick sanity check (no gradient) to verify shapes
    model = SEInceptionResNetV2(in_channels=3, num_classes=3, attention='cbam')
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("output shape:", out.shape)  # expect [2, 3]
