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
import pandas as pd
import re
import gc
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader
from model import *
from datasete import TestDataset
# from dataset import EnhanceDataset
# CUDA_VISIBLE_DEVICES=1 python train.py
class Config:
    model_pre = False
    cata = "all"
   # cata = "STY"
    # cata = "VSTY_ViolentTY"
    check_path = '/mnt/dqdisk/CODE/Tynew/TCIE/all/checkpoints2025-06-12T17:29:44.421201/model70.pth'
    test_dir = "/mnt/dqdisk/Data/Tynew/Test/224/" + cata
    train_label_pkl = "/mnt/dqdisk/Data/Tynew/Train/minmax.pkl"
    # val_label_pkl = "/mnt/dqdisk/Data/Tynew/Val/" + cata + ".pkl"
    # test_label_pkl = "/mnt/dqdisk/Data/Tynew/Test/" + cata + ".pkl"


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

def test(model, dataloader, device, scaler):
    model.eval()
    filenames = []
    predictions = []
    targets_list = []
    
    with torch.no_grad():
        for batch in dataloader:  # 假设返回 (inputs, targets, filenames)
            # 根据实际数据集返回值调整解包方式
            inputs, targets, batch_filenames = batch
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)

            targets_cpu = targets.cpu().numpy()
            outputs_cpu = outputs.cpu().numpy()
            
            pred_target = scaler.inverse_transform(targets_cpu)
            pred_output = scaler.inverse_transform(outputs_cpu)
            
            predictions.append(pred_output)
            targets_list.append(pred_target)
            filenames.extend(batch_filenames)  # 假设返回的是文件名列表

    predictions = np.concatenate(predictions, axis=0)
    targets_list = np.concatenate(targets_list, axis=0)
    
    return filenames, predictions, targets_list

def main():
    scaler = joblib.load(Config.train_label_pkl)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = None

    file_dir = './output_2physical_radiation.txt'
    with open(file_dir, "w") as file:
        #model_path = Config.check_path
        for i in range(96, 97):
            print(i)
            if model is not None:
                del model
                gc.collect()
                if torch.cuda.is_available():  # 如果使用GPU，清理CUDA缓存
                    torch.cuda.empty_cache()
            model = TCIE_swinmae().to(device)
            model_path = f'/mnt/dqdisk/CODE/Tynew/TCIE/all/checkpoints2025-07-20T13:46:39.631745/model{i}.pth'
            # model_path = f'./model{i}.pth'
            model.load_state_dict(torch.load(model_path))
            # test 
            test_folder = Config.test_dir
            #test_dataset = EnhanceDataset(test_folder,is_train=False)
            test_dataset = TestDataset(test_folder, scaler)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            filenames, predictions, targets = test(model, test_loader, device, scaler)
            
            #print("Predictions:", predictions)
            #print("Targets:", targets)
            
            
            true_columns = [f'true_{i}' for i in range(targets.shape[1])]
            pred_columns = [f'pred_{i}' for i in range(predictions.shape[1])]
            
            df = pd.DataFrame(
                np.hstack([
                    np.array(filenames).reshape(-1, 1),  # 文件名列
                    targets,
                    predictions
                ]),
                columns=['filename'] + true_columns + pred_columns
            )
            
            df = df.sort_values(by='filename')

            csv_path = './predictions_my_test.csv'
            df.to_csv(csv_path, index=False)
            print(f"预测结果已保存至：{csv_path}") 

            mae_per_dimension = np.mean(np.abs(predictions - targets), axis=0)
            rmse = np.sqrt(((predictions - targets) ** 2).mean())

            #print(mae_per_dimension)
            #print(rmse)
        
            file.write("epoch: " + str(i) + "\n")
            file.write("Mae: " + str(mae_per_dimension) + "\n")
            file.write("Rmse: " + str(rmse) + "\n")
        
            # file_dir = os.path.dirname(Config.check_path) + '/' + 'output.txt'
            # with open(file_dir, "w") as file:
            #     for i in range(0, len(predictions)):
            #         file.write("Predictions: " + str(predictions[i]) + "\n")
            #         file.write("Targets: " + str(targets[i]) + "\n")
                
            #     file.write("Mae: " + str(mae_per_dimension) + "\n")
            #     file.write("Rmse: " + str(rmse) + "\n")

if __name__ == '__main__':
    main()
