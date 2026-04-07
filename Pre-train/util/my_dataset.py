# This is my dataset for thermal image data
import os
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np

class ThermalDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = os.listdir(root)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.images[index])
        img = Image.open(img_path).convert('RGB')
        # img = np.array(img)
        # img = torch.tensor(img, dtype=torch.float32)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)