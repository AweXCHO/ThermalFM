import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from timm_swin_mae import SwinMAE
from models_infmae_skip4 import MaskedAutoencoderInfMAE
from models_mae import MaskedAutoencoderViT

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