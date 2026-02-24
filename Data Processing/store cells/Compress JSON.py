import cv2
import numpy as np
import json
import os
from pathlib import Path


def simplify_contours(input_path, output_path, epsilon=1.0):
    """
    递归处理文件夹中的JSON文件，简化contour并保存到输出文件夹。

    Args:
        input_path (str): 输入文件夹路径
        output_path (str): 输出文件夹路径
        epsilon (float): Douglas-Peucker算法的简化阈值
    """
    # 确保输出文件夹存在
    os.makedirs(output_path, exist_ok=True)

    # 递归遍历输入文件夹
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.endswith('.json'):
                input_file = os.path.join(root, file)

                # 计算相对路径并在输出文件夹中创建相同结构
                rel_path = os.path.relpath(root, input_path)
                output_dir = os.path.join(output_path, rel_path)
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, file)

                try:
                    # 读取JSON文件
                    with open(input_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # 处理每个核的contour
                    if 'nuc' in data:
                        for nuc_id, nuc in data['nuc'].items():
                            if 'contour' in nuc:
                                # 将contour转换为OpenCV格式
                                contour = np.array(nuc['contour'], dtype=np.int32).reshape(-1, 1, 2)
                                # 应用Douglas-Peucker算法简化
                                approx = cv2.approxPolyDP(contour, epsilon=epsilon, closed=True)
                                # 更新contour为简化后的点列表
                                nuc['contour'] = approx.reshape(-1, 2).tolist()

                    # 保存优化后的JSON到输出文件夹
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, separators=(',', ':'))

                    print(f"Processed and saved: {output_file}")

                except Exception as e:
                    print(f"Error processing {input_file}: {str(e)}")


if __name__ == "__main__":
    # 输入和输出文件夹路径
    input_folder = r"I:\json4"
    output_folder = r"I:\json4-压缩"

    # 调用函数处理文件夹
    simplify_contours(input_folder, output_folder, epsilon=2.0)