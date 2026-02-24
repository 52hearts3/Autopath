import json
import numpy as np
import cv2
import os

def generate_colored_mask(json_path, output_folder, final_size=512):
    # 读取 JSON 文件
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 创建一个 2048x2048 的彩色掩码图像（RGB）
    mask = np.zeros((2048, 2048, 3), dtype=np.uint8)

    # 定义不同类型的颜色（BGR 格式）
    type_colors = {
        0: (255, 255, 255),  # nolabe: 白色 (避免与背景混淆)   无标签
        1: (255, 0, 0),      # neopla: 蓝色  肿瘤
        2: (0, 255, 0),      # inflam: 绿色   炎症
        3: (0, 0, 255),      # connec: 红色   结缔组织
        4: (255, 255, 0),    # necros: 黄色  坏死  (255, 255, 0)
        5: (0, 165, 255)     # no-neo: 橙色 (修正为 BGR 格式的橙色)  非肿瘤
    }

    # 获取核的标注数据
    nuclei = data.get('nuc', {})

    # 为每个核绘制轮廓
    for nuc_id, nuc_data in nuclei.items():
        contour = nuc_data.get('contour', [])
        nuc_type = nuc_data.get('type', 3)  # 默认类型为3 (connec)
        if not contour:
            continue

        # 将轮廓点转换为 NumPy 数组
        contour_points = np.array(contour, dtype=np.int32)

        # 获取该核类型的颜色，默认为红色（如果类型未在type_colors中定义）
        color = type_colors.get(nuc_type, (0, 0, 255))

        # 绘制填充的多边形（彩色掩码）
        cv2.fillPoly(mask, [contour_points], color=color)

    # Resize 掩码到 512x512 像素
    mask_resized = cv2.resize(mask, (final_size, final_size), interpolation=cv2.INTER_AREA)

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 生成输出文件名（将 .json 替换为 .png）
    output_filename = os.path.splitext(os.path.basename(json_path))[0] + '.png'
    output_path = os.path.join(output_folder, output_filename)

    # 保存 resize 后的掩码图像
    cv2.imwrite(output_path, mask_resized)
    print(f"已保存掩码: {output_path}")

def process_json_files_recursive(input_folder, output_folder, final_size=512):
    # 确保输出根文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件和子文件夹
    for item in os.listdir(input_folder):
        input_path = os.path.join(input_folder, item)

        # 如果是文件夹，递归处理
        if os.path.isdir(input_path):
            process_json_files_recursive(input_path, output_folder, final_size)
        # 如果是 JSON 文件，生成掩码
        elif item.lower().endswith('.json'):
            try:
                generate_colored_mask(input_path, output_folder, final_size)
            except Exception as e:
                print(f"处理 {input_path} 时出错: {e}")
        else:
            print(f"跳过非 JSON 文件: {input_path}")

if __name__ == "__main__":
    # 输入和输出文件夹路径
    input_folder = r"I:\json3"  # 替换为你的输入文件夹路径
    output_folder = r"I:\ddpm3\mask" # 替换为你的输出文件夹路径
    process_json_files_recursive(input_folder, output_folder)