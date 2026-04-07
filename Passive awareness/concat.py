import cv2
import os
import numpy as np

def add_text_header(img, text, font_scale=0.8, thickness=2):
    """给图片添加一个白色标题栏并写入黑色文字"""
    h, w = img.shape[:2]
    header_h = 40
    header = np.full((header_h, w, 3), 255, dtype=np.uint8) # 白色背景
    
    # 文字居中计算
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = max((w - text_w) // 2, 0)
    y = (header_h + text_h) // 2 - 5
    
    cv2.putText(header, text, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return np.vstack((header, img))

def merge_pred_and_gt(folder_a, folder_b, output_dir):
    # 1. 准备输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. 获取文件夹A中所有包含 "pred" 的图片
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
    files_a = [f for f in os.listdir(folder_a) if "pred" in f and f.lower().endswith(valid_exts)]
    
    print(f"在文件夹 A 中找到 {len(files_a)} 个包含 'pred' 的文件，开始处理...")

    count = 0
    for f_pred in files_a:
        # --- 构造文件名 ---
        # 假设替换规则是: 把 "pred" 替换为 "GT"
        # 例如: image_pred.png -> image_GT.png
        f_gt = f_pred.replace("pred", "GT")
        
        # --- 构造4个文件的完整路径 ---
        path_a_pred = os.path.join(folder_a, f_pred)
        path_a_gt   = os.path.join(folder_a, f_gt)
        path_b_pred = os.path.join(folder_b, f_pred) # 假设文件夹B也是同样的文件名
        path_b_gt   = os.path.join(folder_b, f_gt)   # 假设文件夹B也有GT

        # --- 检查文件是否存在 ---
        if not os.path.exists(path_b_pred):
            print(f"[跳过] 文件夹 B 中缺少: {f_pred}")
            continue
            
        # 宽容处理：如果某个文件夹里没有GT，我们用黑色空白代替，防止程序崩溃
        # 但通常应该都有
        
        # --- 读取图片 ---
        img_a_pred = cv2.imread(path_a_pred)
        img_a_gt   = cv2.imread(path_a_gt)
        img_b_pred = cv2.imread(path_b_pred)
        img_b_gt   = cv2.imread(path_b_gt)

        if img_a_pred is None: continue

        # 获取基准尺寸 (以图A的Pred为准)
        base_h, base_w = img_a_pred.shape[:2]

        # 定义一个辅助函数来处理缺失图片或缩放
        def process_img(img, label):
            if img is None:
                # 如果读取失败或文件不存在，生成黑色块
                img = np.zeros((base_h, base_w, 3), dtype=np.uint8)
                label += " (Missing)"
            else:
                # 强制缩放到基准尺寸
                if img.shape[:2] != (base_h, base_w):
                    img = cv2.resize(img, (base_w, base_h))
            return add_text_header(img, label)

        # --- 处理四张图 (添加标题 + 缩放) ---
        # 这里你可以修改标题内容，使其更符合你的项目
        # 获取文件夹的名称作为标题的一部分
        name_a = os.path.basename(folder_a)
        name_b = os.path.basename(folder_b)

        vis_a_pred = process_img(img_a_pred, f"{name_a} Pred")
        vis_a_gt   = process_img(img_a_gt,   f"{name_a} GT")
        vis_b_pred = process_img(img_b_pred, f"{name_b} Pred")
        vis_b_gt   = process_img(img_b_gt,   f"{name_b} GT")

        # --- 拼接 ---
        # 布局策略: 2x2 网格
        # [ A_Pred ] [ A_GT ]
        # [ B_Pred ] [ B_GT ]
        
        gap = 5
        # 1. 拼第一行 (Folder A)
        h_sep = np.full((vis_a_pred.shape[0], gap, 3), 255, dtype=np.uint8)
        row1 = np.hstack((vis_a_pred, h_sep, vis_a_gt))

        # 2. 拼第二行 (Folder B)
        row2 = np.hstack((vis_b_pred, h_sep, vis_b_gt))

        # 3. 垂直拼在一起
        v_sep = np.full((gap, row1.shape[1], 3), 255, dtype=np.uint8)
        final_result = np.vstack((row1, v_sep, row2))

        # --- 保存 ---
        save_path = os.path.join(output_dir, f_pred) # 保存名和 pred 文件名一致
        cv2.imwrite(save_path, final_result)
        
        count += 1
        if count % 10 == 0:
            print(f"已处理 {count} 张图片...")

    print(f"全部完成！共生成 {count} 张对比图，保存在: {output_dir}")

# --- 配置与运行 ---
if __name__ == "__main__":
    # 请修改为你的实际路径
    # 文件夹 A (例如 Model 1)
    dir_a = r"./supervised_check_crop_mask2former_visualized" 
    
    # 文件夹 B (例如 Model 2)
    dir_b = r"./supervised_check_crop_originmethod_visualized"
    
    # 结果保存路径
    save_dir = r"./concat_visualized"

    merge_pred_and_gt(dir_a, dir_b, save_dir)