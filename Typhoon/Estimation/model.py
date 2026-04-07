import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from timm_swin_mae import SwinMAE
from models_infmae_skip4 import MaskedAutoencoderInfMAE
from models_mae import MaskedAutoencoderViT
# 按照原文的网络
class TCIE_origin(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SwinMAE()

        self.fc1 = nn.Linear(1024 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 16)
        self.fc3 = nn.Linear(16, 1)
        
    def forward(self, x):
        x = self.encoder(x)
        # (B, 1024, 7, 7)
        x = x[-1]
        x = torch.flatten(x, 1)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

# infmae
class TCIE_infmae(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MaskedAutoencoderInfMAE(
            img_size=[224, 56, 28], patch_size=[4, 2, 2], embed_dim=[256, 384, 768], depth=[2, 2, 11], num_heads=12,
            decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
            mlp_ratio=[4, 4, 4])
        
        self.proj = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.proj:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='leaky_relu')
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=1)
        #print(f"Feature mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
        return self.proj(x)

# origin mae
class TCIE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MaskedAutoencoderViT(
            patch_size=16, embed_dim=1024, depth=24, num_heads=16,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4)
        
        self.proj = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.proj:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='leaky_relu')
    
    def forward(self, x):
        x = self.encoder(x)
        #print(x.shape)
        x = x.mean(dim=1)
        #print(f"Feature mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
        return self.proj(x)
    

class TCIE_swinmae(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SwinMAE()
        
        self.proj = nn.Sequential(
            nn.Linear(1024*7*7, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.proj:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='leaky_relu')
    
    def forward(self, x):
        x = self.encoder(x)[-1]
        
        #print(f"Feature mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
        x = torch.flatten(x, 1)
        return self.proj(x)
    

if __name__ == '__main__':
    net = TCIE()
    input = torch.randn(1, 3, 224, 224)
    output = net(input)
    print(output.shape)