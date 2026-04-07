import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
import argparse
import random
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from model import *
from dataset import ImageDataset
from torch.optim.lr_scheduler import StepLR
from origin_model import SEInceptionResNetV2

class Config:
    seed = 40
    weight_decay = 0.0005
    total_epochs = 50
    init_lr = 0.001  # the origin code is 0.1 and every 50 epochs reduced 0.1 times
    image_size = 224
    img_channels = 3
    class_num = 3
    model_pre = True
    train_dir = "/mnt/dqdisk/Data/Tynew/Train/224"
    val_dir =  "/mnt/dqdisk/Data/Tynew/Val/224"
    test_dir =  "/mnt/dqdisk/Data/Tynew/Test/224"
    batch_size = 64
    model_path = 'checkpoints_resnet/model_epoch42.pth'

def calculate_metrics(all_targets, all_preds, class_names=None):
    """
    计算混淆矩阵和各类评估指标
    """
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(len(np.unique(all_targets)))]
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    
    # 计算各类别的精确率、召回率、F1分数
    precision = precision_score(all_targets, all_preds, average=None, zero_division=0)
    recall = recall_score(all_targets, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)
    
    # 计算宏平均和加权平均
    macro_precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    weighted_precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    weighted_recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    # 打印详细报告
    print("\n" + "="*50)
    print("分类评估报告")
    print("="*50)
    print(f"混淆矩阵:\n{cm}")
    print("\n各类别指标:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: 精确率(P)={precision[i]:.4f}, 召回率(R)={recall[i]:.4f}, F1={f1[i]:.4f}")
    
    print(f"\n宏平均: 精确率={macro_precision:.4f}, 召回率={macro_recall:.4f}, F1={macro_f1:.4f}")
    print(f"加权平均: 精确率={weighted_precision:.4f}, 召回率={weighted_recall:.4f}, F1={weighted_f1:.4f}")
    
    return {
        'confusion_matrix': cm,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }

def train_model(model, train_loader, test_loader, device, checkpointsdir):
    writer = SummaryWriter(log_dir=checkpointsdir)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(
        model.parameters(),
        lr=Config.init_lr,
        weight_decay=Config.weight_decay
    )

    scheduler = StepLR(
        optimizer, 
        step_size=20,
        gamma=0.1
    )

    for epoch in range(Config.total_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} | Current LR: {current_lr:.6f}")

        for batch_idx, (data, target, _) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item():.4f}')
        scheduler.step()

        avg_train_loss = train_loss / total
        train_acc = 100. * correct / total
        test_loss, acc, test_metrics = evaluate(model, test_loader, device, criterion)
        
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Train_Acc: {train_acc:.2f}%")
        print(f"Epoch {epoch} | Test Loss: {test_loss:.4f} | Acc: {acc:.2f}%")

        # 记录到TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', acc, epoch)
        
        # 记录评估指标
        writer.add_scalar('Metrics/macro_precision', test_metrics['macro_precision'], epoch)
        writer.add_scalar('Metrics/macro_recall', test_metrics['macro_recall'], epoch)
        writer.add_scalar('Metrics/macro_f1', test_metrics['macro_f1'], epoch)
        writer.add_scalar('Metrics/weighted_f1', test_metrics['weighted_f1'], epoch)

        # 保存检查点
        torch.save(model.state_dict(), os.path.join(checkpointsdir, f'model_epoch{epoch}.pth'))

def evaluate(model, test_loader, device, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            
            # 收集预测和真实标签
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    acc = 100. * correct / len(test_loader.dataset)
    avg_loss = total_loss / len(test_loader)
    
    # 计算评估指标
    metrics = calculate_metrics(all_targets, all_preds)
    
    return avg_loss, acc, metrics

def test(model, test_loader, device):
    model.eval()
    correct = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            
            # 收集预测和真实标签
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    acc = 100. * correct / len(test_loader.dataset)
    
    # 计算评估指标
    metrics = calculate_metrics(all_targets, all_preds)
    
    return acc, metrics

def model_classify(model, test_loader, device, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    # three class dictionary
    class_paths = {'TS_STS': [], 'STY': [], 'VSTY_ViolentTY': []}

    with torch.no_grad():
        for data, target, paths in test_loader:  # (data, target, paths)
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            
            # 收集预测和真实标签
            pred_cpu = pred.cpu().numpy()
            target_cpu = target.cpu().numpy()
            all_preds.extend(pred_cpu)
            all_targets.extend(target_cpu)
            
            for p, path in zip(pred_cpu, paths):
                class_paths[p].append(path)

    # 保存分类结果到文件
    for class_id, paths in class_paths.items():
        with open(f'class{class_id}.txt', 'w') as f:
            f.write('\n'.join(paths) + '\n')
    
    acc = 100. * correct / len(test_loader.dataset)
    avg_loss = total_loss / len(test_loader)
    
    # 计算评估指标
    metrics = calculate_metrics(all_targets, all_preds)
    
    return avg_loss, acc, metrics

def main():
    seed = Config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # PyTorch GPU（所有设备）
    torch.backends.cudnn.deterministic = True  # 确保cuDNN的确定性
    torch.backends.cudnn.benchmark = False  # 关闭自动优化（可选）

    # 定义数据加载的worker初始化函数
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # 创建随机数生成器
    g = torch.Generator()
    g.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建数据集和数据加载器
    train_dataset = ImageDataset(
        root_dir=Config.train_dir,
        transform=None,
        img_size=Config.image_size
    )
    val_dataset = ImageDataset(
        root_dir=Config.val_dir,
        transform=None,
        img_size=Config.image_size
    )
    test_dataset = ImageDataset(
        root_dir=Config.test_dir,
        transform=None,
        img_size=Config.image_size
    )
    
    train_loader = DataLoader(train_dataset, 
                            batch_size=Config.batch_size,
                            shuffle=True, 
                            worker_init_fn=seed_worker,  # 添加这行
                            generator=g)
    val_loader = DataLoader(val_dataset, 
                            batch_size=Config.batch_size,
                            shuffle=True, 
                            worker_init_fn=seed_worker,  # 添加这行
                            generator=g)
    test_loader = DataLoader(test_dataset,
                           batch_size=Config.batch_size,
                           shuffle=False,
                           worker_init_fn=seed_worker,  # 添加这行
                            generator=g)
    
    model = TCIC_resnet50().to(device)
    
    if Config.model_pre is True:
        model.load_state_dict(torch.load(Config.model_path))

    folder_path = "./checkpoints_resnet"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    
    # train_model(model, train_loader, val_loader, device, folder_path)
    
    # 最终测试
    testacc, test_metrics = test(model, val_loader, device)
    print(f"\n最终测试准确率: {testacc:.2f}%")
    print(f"最终测试宏平均F1: {test_metrics['macro_f1']:.4f}")

if __name__ == '__main__':
    main()