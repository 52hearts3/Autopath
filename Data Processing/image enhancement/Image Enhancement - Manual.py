import os
import json
from PIL import Image
import torch
from torchvision import transforms
import hashlib
import pandas as pd


def load_json_files(folder_path):
    """递归加载文件夹及其子文件夹中的所有 JSON 文件并返回文件路径列表"""
    json_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files


def load_csv_data(csv_path):
    """加载 CSV 文件并返回图像名称与位置信息的映射"""
    df = pd.read_csv(csv_path)
    # 创建图像名称（去除 .png 后缀）到位置信息的映射
    name_to_position = {row["Tile Filename"].replace('.png', ''): row["Position in HE Region"] for _, row in
                        df.iterrows()}
    return name_to_position


def augment_image(input_image_dir, output_dir, json_files, name_to_position):
    """根据 JSON 文件名匹配 CSV 中的位置信息，选择对应图像进行增强，并保存为三通道 RGB 图像"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 定义图像文件与位置的映射
    position_to_image = {
        '右上 (Top-Right)': os.path.join(input_image_dir, '右上 (Top-Right).png'),
        '右下 (Bottom-Right)': os.path.join(input_image_dir, '右下 (Bottom-Right).png'),
        '左上 (Top-Left)': os.path.join(input_image_dir, '左上 (Top-Left).png'),
        '左下 (Bottom-Left)': os.path.join(input_image_dir, '左下 (Bottom-Left).png')
    }

    # 为每个 JSON 文件处理增强
    for json_file in json_files:
        # 获取 JSON 文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(json_file))[0]

        # 检查 CSV 中是否有匹配的图像名称
        if base_name not in name_to_position:
            print(f"跳过 {base_name}：CSV 中没有匹配的条目。")
            continue

        # 获取对应的位置信息
        position = name_to_position[base_name]
        if position not in position_to_image:
            print(f"跳过 {base_name}：无效的位置 {position}。")
            continue

        # 获取对应的图像路径
        input_image_path = position_to_image[position]
        if not os.path.exists(input_image_path):
            print(f"跳过 {base_name}：图像文件 {input_image_path} 不存在。")
            continue

        # 加载原始图像（灰度模式）
        image = Image.open(input_image_path).convert('L')  # 转换为灰度图像

        # 定义增强变换（不包含颜色调整）
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-90, 90)),  # 在 ±90 度范围内随机旋转
            transforms.RandomHorizontalFlip(p=0.5),  # 50% 概率水平翻转
            transforms.RandomVerticalFlip(p=0.5),  # 50% 概率垂直翻转
            transforms.RandomResizedCrop(size=image.size, scale=(0.8, 1.0)),  # 随机裁剪
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 随机平移和缩放
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 随机调整亮度和对比度
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),  # 50% 概率增强锐度
            transforms.Resize((512, 512)),  # 最终统一调整为 512x512 像素
        ])

        # 设置唯一随机种子（基于文件名）
        seed = int(hashlib.md5(base_name.encode()).hexdigest(), 16) % 2 ** 32
        torch.manual_seed(seed)

        # 应用增强变换
        augmented_image = transform(image)

        # 转换为三通道 RGB 图像
        augmented_image_rgb = augmented_image.convert('RGB')

        # 保存增强图像
        output_image_path = os.path.join(output_dir, f"{base_name}.png")
        augmented_image_rgb.save(output_image_path)
        print(f"保存增强后的 RGB 图像：{output_image_path}")


def main():
    # 硬编码参数列表
    input_image_dirs = [
        r"I:\test\分割汇总\W1046896\X24-2092\T",
        r"I:\test\分割汇总\W1046896\X24-2092\U",
        r"I:\test\分割汇总\W1049000\X24-2100\N",
        r"I:\test\分割汇总\W1049000\X24-2100\O",
        r"I:\test\分割汇总\W1049000\X24-2100\P",
        r"I:\test\分割汇总\D6434659\X24-2190\X",
        r"I:\test\分割汇总\D6434659\X24-2190\Y",
        # r"I:\nii_slicer\PART2翻转\PART2增强-分割\X24-14869胡敦根D6965295\A12",
        # r"I:\nii_slicer\PART2翻转\PART2增强-分割\X24-14869胡敦根D6965295\A13",
        # r"I:\nii_slicer\PART2翻转\PART2增强-分割\X24-14869胡敦根D6965295\A15",
        # r"I:\nii_slicer\PART2翻转\PART2增强-分割\X24-14869胡敦根D6965295\A16",
        # r"I:\nii_slicer\PART2翻转\PART2增强-分割\X25-00369邹平W1170580\A11",
        # r"I:\nii_slicer\PART2翻转\PART2增强-分割\X25-00369邹平W1170580\A12",
        # r"I:\nii_slicer\PART2翻转\PART2增强-分割\X25-00369邹平W1170580\A13",
        # r"I:\nii_slicer\PART2翻转\PART2增强-分割\X25-00369邹平W1170580\A14",
        # r"I:\nii_slicer\PART2翻转\PART2增强-分割\X25-00369邹平W1170580\A15",
        # r"I:\nii_slicer\PART2翻转\PART2增强-分割\X25-00369邹平W1170580\A16",
    ]
    csv_path = r"I:\csv\svs_tile_positions_he_batch_all.csv" # CSV 文件路径
    json_folders = [
        r"I:\json2\X24-2092\T",
        r"I:\json2\X24-2092\U",
        r"I:\json2\X24-2100\N",
        r"I:\json2\X24-2100\O",
        r"I:\json2\X24-2100\P",
        r"I:\json2\X24-2190\X",
        r"I:\json2\X24-2190\Y",
        # r"I:\json3\X24-14869\A12",
        # r"I:\json3\X24-14869\A13",
        # r"I:\json3\X24-14869\A15",
        # r"I:\json3\X24-14869\A16",
        # r"I:\json3\X25-00369\A11",
        # r"I:\json3\X25-00369\A12",
        # r"I:\json3\X25-00369\A13",
        # r"I:\json3\X25-00369\A14",
        # r"I:\json3\X25-00369\A15",
        # r"I:\json3\X25-00369\A16",
    ]
    output_dirs = [
        r"I:\CT-T2\W1046896\X24-2092\T",
        r"I:\CT-T2\W1046896\X24-2092\N",
        r"I:\CT-T2\W1049000\X24-2100\N",
        r"I:\CT-T2\W1049000\X24-2100\O",
        r"I:\CT-T2\W1049000\X24-2100\P",
        r"I:\CT-T2\D6434659\X24-2190\X",
        r"I:\CT-T2\D6434659\X24-2190\Y",
        # r"I:\CT-T2增强\D6965295\X24-14869\A12",
        # r"I:\CT-T2增强\D6965295\X24-14869\A13",
        # r"I:\CT-T2增强\D6965295\X24-14869\A15",
        # r"I:\CT-T2增强\D6965295\X24-14869\A16",
        # r"I:\CT-T2增强\W1170580\X25-00369\A11",
        # r"I:\CT-T2增强\W1170580\X25-00369\A12",
        # r"I:\CT-T2增强\W1170580\X25-00369\A13",
        # r"I:\CT-T2增强\W1170580\X25-00369\A14",
        # r"I:\CT-T2增强\W1170580\X25-00369\A15",
        # r"I:\CT-T2增强\W1170580\X25-00369\A16",
    ]

    # 加载 CSV 数据
    name_to_position = load_csv_data(csv_path)
    print(f"已加载 {len(name_to_position)} 条 CSV 数据。")

    # 循环处理每组参数
    for input_image_dir, json_folder, output_dir in zip(input_image_dirs, json_folders, output_dirs):
        print(f"\n处理中：input_image_dir={input_image_dir}, json_folder={json_folder}, output_dir={output_dir}")

        # 加载 JSON 文件
        json_files = load_json_files(json_folder)
        print(f"在 {json_folder} 中找到 {len(json_files)} 个 JSON 文件。")

        # 执行图像增强
        augment_image(input_image_dir, output_dir, json_files, name_to_position)


if __name__ == "__main__":
    main()