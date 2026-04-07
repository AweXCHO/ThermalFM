import scipy.io as sio
import numpy as np
import os

def mat_to_npy(mat_path, npy_path, variable_name=None):
    """
    将 .mat 文件转换为 .npy 文件
    :param mat_path: 输入 .mat 文件路径
    :param npy_path: 输出 .npy 文件路径
    :param variable_name: .mat 文件中具体的变量名（如果为 None，则自动查找）
    """
    try:
        # 1. 读取 .mat 文件
        # mat_contents 是一个字典，key 是变量名，value 是数据
        mat_contents = sio.loadmat(mat_path)
        
        # 2. 提取数据
        data = None
        
        if variable_name:
            # 如果指定了变量名，直接获取
            if variable_name in mat_contents:
                data = mat_contents[variable_name]
            else:
                print(f"错误：在文件 {mat_path} 中找不到变量 '{variable_name}'")
                return
        else:
            # 如果没指定变量名，自动查找第一个非元数据变量
            # .mat 文件包含 '__header__', '__version__' 等元数据，我们要跳过它们
            for key in mat_contents:
                if not key.startswith('__'):
                    data = mat_contents[key]
                    print(f"自动检测到变量名: {key}")
                    break
        
        if data is None:
            print(f"错误：未在 {mat_path} 中找到有效数据")
            return

        # 3. 保存为 .npy
        np.save(npy_path, data)
        print(f"成功转换: {mat_path} -> {npy_path} (Shape: {data.shape})")

    except Exception as e:
        print(f"转换失败: {e}")

# --- 使用示例 ---

# 假设你的文件路径是这样的
input_mat_file = '/mnt/dqdisk/Code/HADAR/TeXNet/hadar/Scene11/GroundTruth/resMap/resMap_R_0004.mat'  # 修改为你的 .mat 文件路径
output_npy_file = '/mnt/dqdisk/Code/HADAR/TeXNet/hadar/Scene11/GroundTruth/resMap/resMap_R_0004.npy' # 输出路径

# 执行转换
mat_to_npy(input_mat_file, output_npy_file)

# 如果你知道 .mat 里变量名具体叫 'resMap' 或 'xMap'，可以指定：
# mat_to_npy(input_mat_file, output_npy_file, variable_name='resMap')