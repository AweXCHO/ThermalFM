import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, img_size=224):
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
        self.classes = ["TS_STS", "STY", "VSTY_ViolentTY"]  # 获取排序后的类别目录
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}  # 创建类别到索引的映射
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        img = Image.open(img_path).convert("RGB")
        
        #img = transforms.Resize((self.img_size, self.img_size))(img)
        
        if self.transform:
            img = self.transform(img)
            
        img_tensor = transforms.ToTensor()(img)
        
        return img_tensor, torch.tensor(label, dtype=torch.long), img_path

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomAffine(
        degrees=0, 
        translate=(0.1, 0.1),
        scale=(0.9, 1.1))
])

test_transform = transforms.Compose([
])
