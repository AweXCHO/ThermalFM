import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

train_label_dir = "/mnt/dqdisk/Data/Tynew/19812013/"
label_name = train_label_dir + "minmax.pkl"

train_labels = []

for filename in os.listdir(train_label_dir):
    if filename.endswith('.jpg'):
        file_name = filename.split('.')[0]  
        reg_values = file_name.split('_')[2]
        label = float(reg_values)  # 假设标签是单数值
        train_labels.append(label)
print(min(train_labels))
# 转换为numpy数组并计算scaler
train_labels = np.array(train_labels).reshape(-1, 1)  # 必须为二维数组
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_labels)  # 计算均值和标准差

# 保存scaler到文件（方便后续加载）
import joblib
joblib.dump(scaler, label_name)