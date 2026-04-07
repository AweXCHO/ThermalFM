import pandas as pd
import numpy as np

# 读取CSV文件（假设文件名为predictions.csv）
df = pd.read_csv("predictions_my_test.csv")

# 定义平滑函数
def apply_smoothing(series):
    smoothed = []
    for t in range(len(series)):
        if t < 2:
            # 前两个时间步保持原值或设为NaN
            smoothed.append(np.nan)
        else:
            # 应用平滑公式：0.49*当前 + 0.29*前1步 + 0.22*前2步
            smoothed_val = 0.49 * series[t] + 0.29 * series[t-1] + 0.22 * series[t-2]
            smoothed.append(smoothed_val)
    return smoothed

# 对每个预测列进行平滑处理
pred_columns = [col for col in df.columns if col.startswith('pred_')]

for col in pred_columns:
    # 生成新的列名
    smooth_col = f'smooth_{col}'
    df[smooth_col] = apply_smoothing(df[col].values)

# 计算评估指标（排除前两行）
valid_rows = df.iloc[2:]

# 计算各维度MAE和总体RMSE
mae_results = {}
rmse_results = []

for col in pred_columns:
    smooth_col = f'smooth_{col}'
    true_col = col.replace('pred_', 'true_')
    
    # 计算MAE
    mae = np.mean(np.abs(valid_rows[smooth_col] - valid_rows[true_col]))
    mae_results[col] = mae
    
    # 计算RMSE分量
    rmse_component = (valid_rows[smooth_col] - valid_rows[true_col])**2
    rmse_results.append(rmse_component)

# 合并所有预测列的RMSE
total_rmse = np.sqrt(pd.concat(rmse_results, axis=1).mean().mean())

# 输出结果
print("MAE per dimension:")
for col, mae in mae_results.items():
    print(f"{col}: {mae:.4f}")

print(f"\nOverall RMSE: {total_rmse:.4f}")

# 保存结果到新文件
output_path = "smoothed_my_test.csv"
df.to_csv(output_path, index=False)
print(f"\n结果已保存至: {output_path}")