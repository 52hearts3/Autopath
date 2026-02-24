import os
from PIL import Image
import pathlib


def split_image(image_path, output_dir):
    """将单张图像分割为四等份，调整大小到512x512并保存"""
    # 打开图像
    img = Image.open(image_path)
    width, height = img.size

    # 计算四等份的边界
    half_width = width // 2
    half_height = height // 2

    # 定义四个区域的坐标和输出文件名
    regions = {
        "左上 (Top-Left)": (0, 0, half_width, half_height),
        "右上 (Top-Right)": (half_width, 0, width, half_height),
        "左下 (Bottom-Left)": (0, half_height, half_width, height),
        "右下 (Bottom-Right)": (half_width, half_height, width, height)
    }

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 分割并保存每个区域
    for region_name, (left, top, right, bottom) in regions.items():
        # 裁剪图像
        region = img.crop((left, top, right, bottom))
        # 调整大小到512x512
        region_resized = region.resize((512, 512), Image.Resampling.LANCZOS)
        # 构造输出文件名
        output_filename = f"{region_name}.png"
        output_path = os.path.join(output_dir, output_filename)
        # 保存裁剪并调整大小后的图像
        region_resized.save(output_path)
        print(f"已保存: {output_path}")


def process_nii_slices(input_dir, output_base_dir):
    """递归处理指定文件夹及其子文件夹中的所有 PNG 文件，并保存到指定的输出文件夹结构"""
    input_path = pathlib.Path(input_dir)
    output_base_path = pathlib.Path(output_base_dir)

    def recursive_process(current_dir):
        """递归遍历文件夹并处理 PNG 文件"""
        for item in current_dir.rglob("*.png"):
            if item.is_file():
                # 创建对应的输出子文件夹，保持与输入相同的结构
                relative_path = item.parent.relative_to(input_path)
                output_subdir = output_base_path / relative_path
                split_image(item, output_subdir)

    # 开始递归处理
    recursive_process(input_path)


if __name__ == "__main__":
    # 输入和输出文件夹路径
    input_directory = r"G:\分析测试\第一批大切片\配准MRI翻转"
    output_directory = r"G:\分析测试\第一批大切片\配准MRI分割"# 自定义输出路径
    process_nii_slices(input_directory, output_directory)
    print("所有图像处理完成！")