import utils
import torch
import modules
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# import model_swin_origin
# import model_vit_origin
import ThermalFM
# import models_infmae_skip4
# import models_mae
# import Dinov3


# Turbulence measurement module in PBCL
class TMM(nn.Module):
    def __init__(self, para, device):
        super().__init__()
        self.para = para
        self.device = device
        self.pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.conv1 = modules.conv1x1_3d(3, 3)
        self.CNN = modules.CNN3D(para, 3, 3, 1 * para.n_feats, stride=(1, 2, 2))
        self.CNN2 = nn.Sequential(
            modules.conv1x1((para.frame_length - 4) * para.n_feats, para.n_feats),
            modules.conv5x5(para.n_feats, para.n_feats, stride=2),
            modules.conv3x3(para.n_feats, 5, stride=2),
            nn.Sigmoid()
        )

    def forward(self, input_data, restored_data=None):  # (BS, F, C, H, W)
        x0 = utils.prepare(False, True, input_data)[:, 2:13]
        if restored_data is None:
            x1 = x0.clone()
        else:
            x1 = utils.prepare(False, True, restored_data)

        s_reference = self.pool(x0[:, :, 0]).unsqueeze(1)
        c1 = x0
        c2 = abs(x0 - x1)
        c3 = abs(x0 - s_reference)  # [2,11,224,224]

        x = torch.cat([c1, c2, c3], dim=2).permute(0, 2, 1, 3, 4)  # (BS, 3, 11, 64, 64)
        x1 = self.conv1(x)
        f = self.CNN(x1)  # (BS, 16, 11, 64, 64)
        f = rearrange(f, 'b c f h w -> b (c f) h w')  # (BS, 16x11, 64, 64)
        y = self.CNN2(f)
        return y


# class DynamicLayerNorm(nn.Module):
#     def __init__(self, normalized_dims=(1, 2, 3, 4), eps=1e-6):
#         super().__init__()
#         self.normalized_dims = normalized_dims  # 默认对 [C, D, H, W] 归一化
#         self.eps = eps

#     def forward(self, x):
#         # 计算需要归一化的维度的均值与方差
#         mean = x.mean(dim=self.normalized_dims, keepdim=True)
#         std = x.std(dim=self.normalized_dims, keepdim=True)
#         return (x - mean) / (std + self.eps)

repo_dir_default = "/home/ayt/PBCL_TMM+MAE/code/dinov3_source_"	# 这里写的是dinov3的源码位置
model_vit_default = "dinov3_convnext_large"	# 这里是指定用哪个模型，具体参考dinov3的readme。
weight_vit_path_default = "/home/ayt/PBCL_TMM+MAE/data/checkpoints/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth"	# 这里是模型的权重文件路径

# Turbulence inhibition module in PBCL
class TIM(nn.Module):
    def __init__(self, para, device):
        super().__init__()
        self.para = para
        self.device = device
        self.neighbors = para.neighboring_frames

        # our mae
        self.MAE = timm_swin_physical.SwinMAE_physical()
        self.MCN = MultiScaleFusion(para)
        # infmae
        # self.infmae = models_infmae_skip4.MaskedAutoencoderInfMAE()
        # self.infmae_converter = TokenToFeatureMap(para)
        # ori mae
        # self.orimae = models_mae.MaskedAutoencoderViT()
        # self.orimae_converter = TokenToFeatureMap(para, in_dim=1024)
        # dinov3
        # self.dinov3 = Dinov3. DinoV3convl16(		# 可以修改参数使用不同的模型
        #     model = model_vit_default,		# 指定模型
        #     weight_path = weight_vit_path_default# 指定权重文件路径
        # )
        # self.MCN_dino = MultiScaleFusion_dino(para)

        self.TST = modules.TST_module(para)
        self.CNN = modules.CNN3D(para, 4, 1, 5 * para.n_feats, stride=(1, 2, 2))
        self.reconstructor = modules.Reconstructor(para)
        self.pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.TS_conv = nn.Sequential(
            nn.ConvTranspose2d(5, 16, kernel_size=1, stride=2, padding=0, output_padding=1),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, input_data, TS_map=None):
        B, _, _, H, W = input_data.shape
        if TS_map is None:
            TS_map = torch.ones([B, 1, H // 4, W // 4]).to(self.device)
        else:
            TS_map = self.TS_conv(TS_map)
        x0 = utils.prepare(False, True, input_data)  # (BS, F, C, H, W) [2, 15, 1, 448, 448]


        # # our mae直接替换
        x = torch.stack([self.MCN(self.MAE(x.repeat(1, 3, 1, 1))) for x in x0.unbind(dim=1)], dim=2) # (BS, 80, F, H/4, W/4) [2, 80, 15, 112, 112]
        
        # infmae直接替换
        # x = torch.stack([self.infmae_converter(self.infmae(x.repeat(1, 3, 1, 1))) for x in x0.unbind(dim=1)], dim=2)

        # orimae直接替换
        # x = torch.stack([self.orimae_converter(self.orimae(x.repeat(1, 3, 1, 1))) for x in x0.unbind(dim=1)], dim=2)

        # dinov3直接替换
        # x = torch.stack([self.MCN_dino(self.dinov3(x.repeat(1, 3, 1, 1))) for x in x0.unbind(dim=1)], dim=2)

        s_reference = self.pool(x)[:, :, 0]  # (BS, 80, H/4, W/4) [2, 80, 112, 112]
        batch_size, channels, frames, _, _ = x.shape
        after_cnn, outputs = [], []
        for i in range(frames):
            after_cnn.append(x[:, :, i, :, :])
        for i in range(self.neighbors, frames - self.neighbors):
            out = self.TST(after_cnn[i-self.neighbors: i+self.neighbors+1], s_reference, TS_map)
            out = self.reconstructor(out)
            out = out + x0[:, i, :, :, :]
            outputs.append(out.unsqueeze(dim=1))
        res_out = utils.prepare_reverse(False, True, torch.cat(outputs, dim=1))
        # print(res_out.shape)
        return res_out

class TokenToFeatureMap(nn.Module):
    def __init__(self, para, in_dim=768):
        super(TokenToFeatureMap, self).__init__()
        self.out_channels = 5 * para.n_feats

        # 通道映射：将 768 映射为期望的输出通道数
        self.channel_mapper = nn.Conv2d(in_dim, self.out_channels, kernel_size=1)

        # 两层转置卷积，每层放大 2×，共放大 4×（14×14 → 56×56）
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(self.out_channels, self.out_channels, kernel_size=2, stride=2),  # 14→28
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.out_channels, self.out_channels, kernel_size=2, stride=2),  # 28→56
        )

    def forward(self, x):
        """
        x: Tensor of shape (B, N, C), where N must be a square number (e.g., 14x14=196)
        """
        B, N, C = x.shape
        H_patch = W_patch = int(N ** 0.5)
        assert H_patch * W_patch == N, "Token number N must be a perfect square"
        # 变形为 2D 特征图: (B, C, H, W)
        x = x.view(B, H_patch, W_patch, C).permute(0, 3, 1, 2)
        # 映射通道维度
        x = self.channel_mapper(x)
        # 上采样到 4× 尺寸
        x = self.upsample(x)
        return x

class MultiScaleFusion(nn.Module):
    def __init__(self, para):
        super(MultiScaleFusion, self).__init__()
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 5 * para.n_feats)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)  # H/4 to H/2
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)  # H/2 to H
        self.deconv3 = nn.ConvTranspose2d(5 * para.n_feats, 5 * para.n_feats, kernel_size=2, stride=2)  # H to 2H

    def forward(self, input):
        x0, x1, x2, x3 = input
        # torch.Size([2, 256, 28, 28]) torch.Size([2, 512, 14, 14]) torch.Size([2, 1024, 7, 7]) torch.Size([2, 1024, 7, 7])
        
        # x3 + x2
        x = x3 + x2
        x = self.linear1(x.view(-1, x.size(3))).view(x.size(0), x.size(1), x.size(2), 512)  # (2, H/4, W/4, 512)
        x = self.deconv1(x.permute(0, 3, 1, 2))  # (2, 512, H/2, W/2)
        x = x.permute(0, 2, 3, 1)  # (2, H/2, W/2, 512)
        
        # x + x1
        x = x + x1
        x = self.linear2(x.view(-1, x.size(3))).view(x.size(0), x.size(1), x.size(2), 256)  # 线性变换到256
        x = self.deconv2(x.permute(0, 3, 1, 2))  # (2, 256, H, W)
        x = x.permute(0, 2, 3, 1)  # (2, H, W, 256)

        # x + x0
        x = x + x0
        x = self.linear3(x.view(-1, x.size(3))).view(x.size(0), x.size(1), x.size(2), 80)  # (2, H, W, 80)
        x = self.deconv3(x.permute(0, 3, 1, 2))  # (2, 80, 2H, 2W)
        # print(x.shape)
        return x
    

class MultiScaleFusion_dino(nn.Module):
    def __init__(self, para):
        super(MultiScaleFusion_dino, self).__init__()

        # 线性层替换为1x1卷积，更自然地匹配[N, C, H, W]格式
        self.reduce3 = nn.Conv2d(1536, 768, kernel_size=1)   # x3 -> 768
        self.reduce2 = nn.Conv2d(768, 384, kernel_size=1)    # 融合后 -> 384
        self.reduce1 = nn.Conv2d(384, 192, kernel_size=1)    # 融合后 -> 192
        self.out_conv = nn.Conv2d(192, 5 * para.n_feats, kernel_size=1)

        # 上采样模块（转置卷积）
        self.deconv1 = nn.ConvTranspose2d(768, 768, kernel_size=2, stride=2)   # 7→14
        self.deconv2 = nn.ConvTranspose2d(384, 384, kernel_size=2, stride=2)   # 14→28
        self.deconv3 = nn.ConvTranspose2d(192, 192, kernel_size=2, stride=2)   # 28→56

        self.act = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x0, x1, x2, x3 = inputs  # resolutions: 56, 28, 14, 7
        # torch.Size([2, 192, 56, 56]) torch.Size([2, 384, 28, 28]) torch.Size([2, 768, 14, 14]) torch.Size([2, 1536, 7, 7])

        # ============ Stage 1: fuse x3 + x2 ============
        x3_up = self.deconv1(self.reduce3(x3))  # [B,768,14,14]
        x = x3_up + x2  # 通道匹配: 768

        # ============ Stage 2: fuse with x1 ============
        x = self.act(self.reduce2(x))  # -> [B,384,14,14]
        x = self.deconv2(x)            # -> [B,384,28,28]
        x = x + x1                     # 融合

        # ============ Stage 3: fuse with x0 ============
        x = self.act(self.reduce1(x))  # -> [B,192,28,28]
        x = self.deconv3(x)            # -> [B,192,56,56]
        x = x + x0                     # 融合

        # ============ Stage 4: output & upsample ============
        x = self.act(self.out_conv(x))   # [B,5*n_feats,56,56]
        # x = self.deconv4(x)              # [B,5*n_feats,112,112]
        # print(x.shape)
        return x
