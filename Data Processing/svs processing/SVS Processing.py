import os
import warnings
import openslide
from PIL import Image
import numpy as np
from pathlib import Path as PathLib
import re

# 抑制TIFF警告
warnings.filterwarnings("ignore")

# 1. 读取SVS图像并降采样，生成白区掩码
def read_svs_image(svs_path, target_size=(4096, 4096), white_threshold=200):
    try:
        slide = openslide.OpenSlide(svs_path)
        width, height = slide.dimensions
        downsample = max(width / target_size[0], height / target_size[1])
        level = slide.get_best_level_for_downsample(downsample)
        img = slide.read_region((0, 0), level, slide.level_dimensions[level])
        img = img.convert("RGB")
        img = img.resize(target_size, resample=Image.Resampling.LANCZOS)
        img_np = np.array(img)

        # 生成白区掩码（基于RGB均值阈值）
        gray = np.mean(img_np, axis=2)
        mask = gray < white_threshold  # 小于阈值的为非白区
        scale_x = width / target_size[0]
        scale_y = height / target_size[1]
        print(f"调试：SVS尺寸 ({width}, {height}), 降采样比例 ({scale_x:.2f}, {scale_y:.2f})")
        return img_np, slide, scale_x, scale_y, mask
    except Exception as e:
        print(f"错误：无法读取SVS文件 {svs_path}: {e}")
        return None, None, None, None, None

# 2. 分割小切片并保存到slice文件夹
def extract_tiles(slide, scale_x, scale_y, mask, svs_path, output_dir, tile_size=2048, white_threshold=200):
    width, height = slide.dimensions
    slice_dir = os.path.join(output_dir, 'slice')
    os.makedirs(slice_dir, exist_ok=True)

    tile_count = 0

    # 遍历整个图像，提取切片
    for x in range(0, width, tile_size):
        for y in range(0, height, tile_size):
            # 检查白区
            center_x_scaled = int(x / scale_x)
            center_y_scaled = int(y / scale_y)
            if center_x_scaled >= mask.shape[1] or center_y_scaled >= mask.shape[0]:
                print(f"调试：跳过切片 at ({x}, {y}) in {svs_path}：坐标超出降采样掩码范围")
                continue
            if not mask[center_y_scaled, center_x_scaled]:
                print(f"调试：跳过切片 at ({x}, {y}) in {svs_path}：白区（掩码检查）")
                continue

            # 读取小切片
            try:
                tile = slide.read_region((x, y), 0, (tile_size, tile_size))
                tile = tile.convert("RGB")
                tile_np = np.array(tile)
                tile_gray = np.mean(tile_np, axis=2)
                if np.mean(tile_gray) >= white_threshold:
                    print(f"调试：跳过切片 at ({x}, {y}) in {svs_path}：白区（灰度均值 {np.mean(tile_gray):.2f} >= {white_threshold}）")
                    continue

                # 保存切片
                output_path = os.path.join(slice_dir, f"{os.path.splitext(os.path.basename(svs_path))[0]}_tile_{x}_{y}.png")
                tile.save(output_path)
                tile_count += 1
                print(f"保存切片：{output_path}")
            except Exception as e:
                print(f"警告：无法提取切片 at ({x}, {y}) in {svs_path}: {e}")
                continue

    print(f"完成处理 {svs_path}，共生成 {tile_count} 个切片")
    return tile_count

# 3. 提取编号和层数
def extract_number_and_layer(svs_path):
    # 从路径中提取编号（上一级文件夹名称）
    path_parts = PathLib(svs_path).parts
    number = None
    for part in path_parts:
        if re.match(r'^(?:[A-Z])?\d{2}-\d{5}$', part):  # 匹配编号格式，如 X25-03689 或 25-03689
            number = part
            break
    if not number:
        return None, None

    # 从文件名中提取层数（例如 X25-03689-A14-2025-07-13_20_53_20.svs 中的 A14）
    filename = os.path.basename(svs_path)
    layer_match = re.match(r'^(?:[A-Z])?\d{2}-\d{5}-([A-Z]\d{2})-\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}\.svs$', filename)
    if layer_match:
        layer = layer_match.group(1)
        return number, layer
    return number, None

# 4. 递归查找SVS文件并处理
def find_svs_and_process(folder_a, folder_d, target_size=(4096, 4096), tile_size=2048):
    for root, dirs, files in os.walk(folder_a):
        for file in files:
            if file.endswith('.svs'):
                svs_path = os.path.join(root, file)
                print(f"处理SVS文件：{svs_path}")

                # 提取编号和层数
                number, layer = extract_number_and_layer(svs_path)
                if number is None:
                    print(f"警告：无法从路径 {svs_path} 中提取编号，跳过")
                    continue
                if layer is None:
                    print(f"警告：无法从路径 {svs_path} 中提取层数，跳过")
                    continue

                print(f"提取编号：{number}, 层数：{layer}")

                # 创建输出目录：/Output/编号/层数
                output_dir = os.path.join(folder_d, number, layer)
                os.makedirs(output_dir, exist_ok=True)

                # 读取和处理SVS图像
                image, slide, scale_x, scale_y, mask = read_svs_image(svs_path, target_size)
                if image is None or slide is None or mask is None:
                    continue

                # 提取小切片
                extract_tiles(slide, scale_x, scale_y, mask, svs_path, output_dir, tile_size)
                slide.close()

# 主函数
def main(folder_a, folder_d, target_size=(4096, 4096), tile_size=2048):
    if not os.path.exists(folder_a):
        print(f"错误：文件夹A {folder_a} 不存在")
        return
    if not os.path.exists(folder_d):
        os.makedirs(folder_d, exist_ok=True)
    print(f"开始处理文件夹A: {folder_a}")
    print(f"输出文件夹D: {folder_d}")
    find_svs_and_process(folder_a, folder_d, target_size, tile_size)

# 示例调用
folder_a = r"F:\dataset\The Fourth Batch of Whole Slide Images\batch_six"
folder_d = r"G:\Output2"
main(folder_a, folder_d, target_size=(4096, 4096), tile_size=2048)