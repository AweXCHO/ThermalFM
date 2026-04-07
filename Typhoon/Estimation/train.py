import numpy as np
import sys
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch
import os
import logging
import datetime
import torch.nn.functional as F
import joblib
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader
from model import *
from datasete import CustomDataset
# from dataset import EnhanceDataset
# CUDA_VISIBLE_DEVICES=1 python train.py
class Config:
    total_epochs = 100
    lr = 0.0001  
    model_pre = False
    cata = "all"
    #cata = "TS_STS"
    #cata = "STY"
    # cata = "VSTY_ViolentTY"
    check_path = "/mnt/dqdisk/CODE/Tynew/TCIE/all/checkpoints2025-03-20T18:42:10.031883/model95.pth"
    train_dir = "/mnt/dqdisk/Data/Tynew/Train/224/" + cata
    val_dir =  "/mnt/dqdisk/Data/Tynew/Val/224/" + cata
    test_dir = "/mnt/dqdisk/Data/Tynew/Test/224/" + cata
    train_label_pkl = "/mnt/dqdisk/Data/Tynew/Train/minmax.pkl"
    # val_label_pkl = "/mnt/dqdisk/Data/Tynew/Val/" + cata + ".pkl"
    # test_label_pkl = "/mnt/dqdisk/Data/Tynew/Test/" + cata + ".pkl"
    batch_size = 64
    loss_number = 0 # 0 = smooth, 1 = L1

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device, scaler):
    model.eval()
    running_loss = 0.0
    running_mse = 0.0 
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            
            targets_cpu = targets.cpu().numpy()
            outputs_cpu = outputs.cpu().numpy()
            
            pred_target = scaler.inverse_transform(targets_cpu)
            pred_output = scaler.inverse_transform(outputs_cpu)

            # 转换为tensor计算MSE
            mse = F.mse_loss(torch.tensor(pred_output), 
                            torch.tensor(pred_target))
            running_mse += mse.item()

    return running_loss / len(dataloader), np.sqrt(running_mse / len(dataloader))

def test(model, dataloader, device):
    model.eval()
    predictions = []
    targets_list = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    targets_list = np.concatenate(targets_list, axis=0)
    
    return predictions, targets_list

def main():
    scaler = joblib.load(Config.train_label_pkl)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    folder_path = './' + Config.cata + '/checkpoints' + datetime.datetime.now().isoformat()

    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    
    train_dataset = CustomDataset(Config.train_dir, scaler)
    val_dataset = CustomDataset(Config.val_dir, scaler)
    
    train_loader = DataLoader(train_dataset, Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, Config.batch_size, shuffle=False)
    
    model = TCIE().to(device)

    if Config.model_pre == True:
        model_path = Config.check_path
        model.load_state_dict(torch.load(model_path), strict=False)
    
    if Config.loss_number == 1:
        criterion = nn.L1Loss()
    elif Config.loss_number == 0:
        criterion = nn.SmoothL1Loss() 

    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    
    num_epochs = Config.total_epochs

    logging.basicConfig(filename=Config.cata + datetime.datetime.now().isoformat()+'.txt', level=logging.INFO)
    for epoch in range(num_epochs):
        # train
        train_loss = train(model, train_loader, criterion, optimizer, device)
        
        # val
        val_loss, val_mse = validate(model, val_loader, criterion, device, scaler)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
            f'Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f},'
            f'Validation RMSE: {val_mse:.4f}')
        
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], '
                    f'Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, '
                    f'Validation RMSE: {val_mse:.4f}')
        torch.save(model.state_dict(), folder_path + f'/model{epoch}.pth')
    
    # test 
    test_folder = Config.test_dir
    #test_dataset = EnhanceDataset(test_folder,is_train=False)
    test_dataset = CustomDataset(test_folder, scaler)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    predictions, targets = test(model, test_loader, device)
    
    print("Predictions:", predictions)
    print("Targets:", targets)
    
    mae_per_dimension = np.mean(np.abs(predictions - targets), axis=0)
    with open("output.txt", "w") as file:
        for i in range(0, len(predictions)):
            file.write("Predictions: " + str(predictions[i]) + "\n")
            file.write("Targets: " + str(targets[i]) + "\n")
        
        file.write("Mae: " + str(mae_per_dimension) + "\n")

if __name__ == '__main__':
    main()
