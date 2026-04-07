import scipy.io
import numpy as np
from PIL import Image
import os

def mat_to_image(mat_path, save_path, variable_name=None):
    try:
        # 1. 加载 mat 文件
        mat = scipy.io.loadmat(mat_path)
        
        # 2. 自动寻找变量名 (如果用户没指定，这就取第一个非meta的变量)
        if variable_name is None:
            for key in mat.keys():
                if not key.startswith('__'):
                    variable_name = key
                    break
        
        if variable_name not in mat:
            print(f"错误: 找不到变量名 '{variable_name}'")
            return

        data = mat[variable_name]
        print(f"正在处理变量: {variable_name}, 形状: {data.shape}")

        # 3. 数据归一化 (关键步骤)
        # 确保数据在 0-255 之间并转为 uint8
        if data.dtype != np.uint8:
            # 归一化到 0-1
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
            # 扩展到 0-255
            data = (data * 255).astype(np.uint8)

        # 4. 形状修正
        # MATLAB 有时是 (C, H, W) 或 (H, W, C)，PIL 需要 (H, W, C)
        # 如果是 3 通道且第一个维度是 3 (例如 3x256x256)，进行转置
        if data.ndim == 3 and data.shape[0] == 3:
            data = data.transpose(1, 2, 0)
        
        # 如果有多张图 (例如 N, H, W, C)，这里只存第一张
        if data.ndim == 4:
            print("检测到由多张图片组成的数组，仅保存第一张...")
            data = data[0]

        # 5. 保存
        img = Image.fromarray(data)
        img.save(save_path)
        print(f"成功保存: {save_path}")

    except NotImplementedError:
        print("错误: 请看下面的'特殊情况' (v7.3版本)")
    except Exception as e:
        print(f"发生错误: {e}")

# --- 使用方法 ---
# 假设你的文件叫 data.mat，想保存为 result.png
# 如果你知道变量名是 'img_data'，就传进去，不知道就留空
mat_to_image('hadar/Scene1/HeatCubes/L_0001_heatcube.mat', 'hadar/Scene1/HeatCubes/L_0001_heatcube.png')