import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# ---------------------- 1. 配置基础参数（可根据需求修改）----------------------
# 类别名称（与混淆矩阵行/列顺序一致）
classes = ["TS_STS", "STY", "VSTY_ViolentTY"]
# 你的混淆矩阵数据（行=真实标签，列=预测标签）
confusion_matrix = np.array([
    [929, 78, 3],    # TS_STS 真实标签的预测分布
    [162, 250, 55],  # STY 真实标签的预测分布
    [15, 88, 325] # VSTY_ViolentTY 真实标签的预测分布
])
# 图表样式配置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文（Windows）
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 支持负号

# ---------------------- 2. 绘制混淆矩阵热力图 ----------------------
fig, ax = plt.subplots(figsize=(10, 8))  # 设置图表大小

# 绘制热力图（cmap 可选：Blues、Greens、Reds、viridis 等）
im = ax.imshow(confusion_matrix, cmap="Blues", interpolation="nearest", aspect="auto")

# 添加颜色条（显示数值范围）
cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("样本数量", rotation=-90, va="bottom", fontsize=12)

# 设置坐标轴标签
ax.set(
    xticks=np.arange(confusion_matrix.shape[1]),
    yticks=np.arange(confusion_matrix.shape[0]),
    xticklabels=classes,
    yticklabels=classes,
    xlabel="预测标签",
    ylabel="真实标签",
    title="混淆矩阵（Confusion Matrix）"
)

# 旋转 x 轴标签（避免类别名称过长重叠）
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
plt.setp(ax.get_yticklabels(), fontsize=10)

# ---------------------- 3. 在热力图中添加数值标注 ----------------------
# 遍历混淆矩阵，在每个单元格显示数值
thresh = confusion_matrix.max() / 2  # 数值颜色阈值（区分深色/浅色背景）
for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
        ax.text(
            j, i, confusion_matrix[i, j],
            ha="center", va="center",
            color="white" if confusion_matrix[i, j] > thresh else "black",
            fontsize=9
        )

# 调整布局（防止标签被截断）
fig.tight_layout()

# ---------------------- 4. 计算并打印分类评估指标 ----------------------
# 生成真实标签和预测标签的一维数组（用于计算评估指标）
y_true = []
y_pred = []
for true_label in range(len(classes)):
    for pred_label in range(len(classes)):
        count = confusion_matrix[true_label, pred_label]
        y_true.extend([true_label] * count)
        y_pred.extend([pred_label] * count)

# 打印精确率（Precision）、召回率（Recall）、F1-Score
print("="*50)
print("分类评估指标（Classification Metrics）")
print("="*50)
print(classification_report(
    y_true, y_pred,
    target_names=classes,
    digits=3  # 保留 3 位小数
))

# ---------------------- 5. 显示/保存图表 ----------------------
plt.show()  # 显示图表
fig.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")  # 保存为高清图片（可选）