import os
import re
import csv
import sys
import io
import openslide
from pathlib import Path
from math import ceil
import cv2
import numpy as np
import matplotlib.pyplot as plt

class OutputFilter(io.TextIOBase):
    """
    自定义输出过滤器，拦截 sys.stdout 和 sys.stderr，过滤包含 'Warning' 的输出。
    """
    def __init__(self, original_stream):
        self.original_stream = original_stream
        self.buffer = []

    def write(self, text):
        if 'warning' not in text.lower():
            self.original_stream.write(text)
        self.buffer.append(text)

    def flush(self):
        self.original_stream.flush()

    def __getattr__(self, name):
        return getattr(self.original_stream, name)

def detect_he_region(slide, level=4, threshold_lower=(120, 30, 30), threshold_upper=(180, 255, 255), output_dir="debug_masks"):
    """
    检测SVS图像中的HE染色区域，返回HE区域的边界框 (x_min, y_min, x_max, y_max) 和轮廓中心点。
    使用低分辨率层（默认level=4）以平衡速度和细节，基于颜色阈值分割HE区域，并显示轮廓预览。
    保存掩膜图像到指定目录以便调试。显示新图像前关闭前一图像。
    """
    try:
        # 读取低分辨率图像
        img = slide.read_region((0, 0), level, slide.level_dimensions[level])
        img = np.array(img.convert('RGB'))  # 转换为RGB格式

        # 转换为HSV颜色空间，便于颜色分割
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # 定义HE染色区域的颜色范围（紫色/粉红色调）
        lower_bound = np.array(threshold_lower)
        upper_bound = np.array(threshold_upper)
        mask = cv2.inRange(img_hsv, lower_bound, upper_bound)

        # 形态学处理：去除噪声，连接断续区域
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)  # 开运算去噪
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # 闭运算连接区域

        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"警告：未在SVS文件 {os.path.basename(slide._filename)} 中检测到HE区域，使用全图尺寸")
            # 保存空掩膜以便调试
            os.makedirs(output_dir, exist_ok=True)
            mask_path = os.path.join(output_dir, f"mask_{os.path.basename(slide._filename)}.png")
            cv2.imwrite(mask_path, mask)
            print(f"已保存掩膜图像：{mask_path}")
            return 0, 0, slide.dimensions[0], slide.dimensions[1], None

        # 筛选面积较大的轮廓（面积大于图像面积的1%）
        total_area = img.shape[0] * img.shape[1]
        valid_contours = [c for c in contours if cv2.contourArea(c) > total_area * 0.01]
        print(f"检测到 {len(contours)} 个轮廓，筛选后保留 {len(valid_contours)} 个有效轮廓")

        if not valid_contours:
            print(f"警告：未找到有效HE区域轮廓，使用全图尺寸")
            mask_path = os.path.join(output_dir, f"mask_{os.path.basename(slide._filename)}.png")
            cv2.imwrite(mask_path, mask)
            print(f"已保存掩膜图像：{mask_path}")
            return 0, 0, slide.dimensions[0], slide.dimensions[1], None

        # 合并有效轮廓的边界框
        x_min = y_min = float('inf')
        x_max = y_max = float('-inf')
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        # 计算合并后区域的几何中心
        combined_mask = np.zeros_like(mask)
        cv2.drawContours(combined_mask, valid_contours, -1, 255, thickness=cv2.FILLED)
        moments = cv2.moments(combined_mask)
        if moments['m00'] != 0:
            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])
        else:
            center_x = x_min + (x_max - x_min) // 2
            center_y = y_min + (y_max - y_min) // 2

        # 将低分辨率坐标转换回最高分辨率（level 0）
        downsample = slide.level_downsamples[level]
        x_min = int(x_min * downsample)
        y_min = int(y_min * downsample)
        x_max = int(x_max * downsample)
        y_max = int(y_max * downsample)
        center_x = int(center_x * downsample)
        center_y = int(center_y * downsample)

        # 确保边界框在SVS图像范围内
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(slide.dimensions[0], x_max)
        y_max = min(slide.dimensions[1], y_max)

        # 保存掩膜图像以便调试
        os.makedirs(output_dir, exist_ok=True)
        mask_path = os.path.join(output_dir, f"mask_{os.path.basename(slide._filename)}.png")
        cv2.imwrite(mask_path, mask)
        print(f"已保存掩膜图像：{mask_path}")

        # 关闭前一图像窗口
        plt.close('all')  # 确保关闭所有之前的matplotlib窗口

        # 预览HE区域轮廓和掩膜
        plt.ion()  # 启用交互模式
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(img)
        for contour in valid_contours:
            ax1.plot(contour[:, :, 0], contour[:, :, 1], 'r-', linewidth=1, label='HE区域轮廓' if contour is valid_contours[0] else "")
        ax1.plot(center_x / downsample, center_y / downsample, 'b*', markersize=15, label='轮廓中心')
        ax1.set_title(f"HE区域检测 - {os.path.basename(slide._filename)}")
        ax1.legend()

        ax2.imshow(mask, cmap='gray')
        ax2.set_title(f"掩膜 - {os.path.basename(slide._filename)}")
        plt.show(block=False)
        plt.pause(2)  # 显示2秒后继续

        print(f"检测到HE区域：({x_min}, {y_min}, {x_max}, {y_max})，轮廓中心：({center_x}, {center_y})")
        return x_min, y_min, x_max, y_max, (center_x, center_y)
    except Exception as e:
        print(f"错误：HE区域检测失败：{e}")
        plt.close('all')
        mask_path = os.path.join(output_dir, f"mask_{os.path.basename(slide._filename)}.png")
        cv2.imwrite(mask_path, mask)
        print(f"已保存掩膜图像：{mask_path}")
        return 0, 0, slide.dimensions[0], slide.dimensions[1], None

def get_tile_position(tile_x, tile_y, tile_size, he_region, contour_center):
    """
    根据切片坐标和HE区域轮廓中心点，判断切片的相对位置（不检查是否在HE区域边界框内）。
    """
    # 使用轮廓中心点判断相对位置
    if contour_center is None:
        return "无法确定位置 (No Contour Center)"

    center_x, center_y = contour_center
    tile_center_x = tile_x + tile_size / 2
    tile_center_y = tile_y + tile_size / 2

    if tile_center_x <= center_x and tile_center_y <= center_y:
        return "左上 (Top-Left)"
    elif tile_center_x <= center_x and tile_center_y > center_y:
        return "左下 (Bottom-Left)"
    elif tile_center_x > center_x and tile_center_y <= center_y:
        return "右上 (Top-Right)"
    else:
        return "右下 (Bottom-Right)"

def process_tile_filename(filename, slide_width, slide_height, tile_size=2048):
    """
    从切片文件名中提取前缀和坐标，支持两种格式：
    1. prefix_tile_X_Y.png（如 24-01521N-2025-02-20_23_56_16_tile_2048_26624.png）
    2. prefix_顺序编号.png（如 24-01521O-2025-02-20_23_48_31_000446.png）
    返回 (prefix, x, y, tile_size) 或 None。
    """
    # 尝试匹配坐标格式
    pattern_coord = r'(.+)_tile_(\d+)_(\d+)\.png$'
    match_coord = re.search(pattern_coord, filename)
    if match_coord:
        prefix = match_coord.group(1)
        x, y = int(match_coord.group(2)), int(match_coord.group(3))
        return prefix, x, y, tile_size

    # 尝试匹配顺序编号格式
    pattern_seq = r'(.+)_(\d{06,})\.png$'
    match_seq = re.search(pattern_seq, filename)
    if match_seq:
        prefix = match_seq.group(1)
        seq_num = int(match_seq.group(2))  # 顺序编号，例如 000446 -> 446
        cols = ceil(slide_width / tile_size)  # 每行切片数
        row = seq_num // cols  # 行号
        col = seq_num % cols   # 列号
        x = col * tile_size    # x 坐标
        y = row * tile_size    # y 坐标
        return prefix, x, y, tile_size

    return None

def find_svs_files(svs_folder):
    """
    递归查找所有 SVS 文件，并提取文件名（不含 .svs 后缀）。
    返回 {prefix: svs_path} 字典。
    """
    svs_files = {}
    for root, _, files in os.walk(svs_folder):
        for file in files:
            if file.endswith('.svs'):
                prefix = file[:-4]
                svs_files[prefix] = os.path.join(root, file)
                print(f"找到 SVS 文件：{file}，前缀：{prefix}")
    return svs_files

def match_svs_and_tiles(svs_folder, tile_folder, output_file):
    """
    递归查找 SVS 文件和切片文件，进行精确匹配，并保存结果到 CSV。
    预先计算每个 SVS 文件的 HE 区域，避免重复计算。
    """
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    svs_files = find_svs_files(svs_folder)
    if not svs_files:
        print(f"错误：未在 {svs_folder} 中找到任何 SVS 文件")
        return

    # 预先计算每个 SVS 文件的 HE 区域和轮廓中心
    he_regions = {}
    for prefix, svs_path in svs_files.items():
        try:
            slide = openslide.OpenSlide(svs_path)
            slide_width, slide_height = slide.dimensions
            he_region = detect_he_region(slide, output_dir=os.path.join(output_dir, "debug_masks"))
            he_regions[prefix] = {
                'he_region': he_region[:4],  # (x_min, y_min, x_max, y_max)
                'contour_center': he_region[4],  # (center_x, center_y) or None
                'slide_width': slide_width,
                'slide_height': slide_height
            }
            slide.close()
        except Exception as e:
            print(f"错误：无法读取 SVS 文件 {svs_path}: {e}")
            he_regions[prefix] = {
                'he_region': (0, 0, slide_width, slide_height),
                'contour_center': None,
                'slide_width': slide_width,
                'slide_height': slide_height
            }

    results = []

    for root, _, files in os.walk(tile_folder):
        for file in files:
            if not file.endswith('.png'):
                continue

            # 解析切片文件名
            tile_info = None
            for prefix in svs_files.keys():
                if file.startswith(prefix):
                    try:
                        tile_info = process_tile_filename(
                            file,
                            he_regions[prefix]['slide_width'],
                            he_regions[prefix]['slide_height']
                        )
                        if tile_info:
                            break
                    except Exception as e:
                        print(f"错误：无法解析切片文件名 {file}: {e}")
                        continue

            if not tile_info:
                print(f"警告：切片文件名 {file} 不符合预期格式或未找到对应 SVS 文件，跳过")
                continue

            prefix, tile_x, tile_y, tile_size = tile_info
            if prefix not in he_regions:
                print(f"警告：切片 {file} 的前缀 {prefix} 未找到对应 SVS 文件，跳过")
                continue

            # 获取缓存的HE区域和轮廓中心
            he_region = he_regions[prefix]['he_region']
            contour_center = he_regions[prefix]['contour_center']

            # 计算切片在HE区域内的位置
            position = get_tile_position(tile_x, tile_y, tile_size, he_region, contour_center)

            results.append({
                'tile_filename': file,
                'position': position
            })
            print(f"匹配成功：切片 {file} -> SVS {svs_files[prefix]}，HE区域内位置：{position} (坐标: ({tile_x}, {tile_y}))")

    # 关闭所有 matplotlib 窗口
    plt.close('all')

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Tile Filename', 'Position in HE Region'])
        for result in results:
            writer.writerow([result['tile_filename'], result['position']])

    print("\n统计信息：")
    print(f"- 总计找到 {len(svs_files)} 个 SVS 文件")
    print(f"- 总计匹配 {len(results)} 个切片")
    unmatched_svs = [svs for prefix, svs in svs_files.items() if
                     not any(r['tile_filename'].startswith(prefix) for r in results)]
    if unmatched_svs:
        print("- 未匹配的 SVS 文件：")
        for svs in unmatched_svs:
            print(f"  - {svs}")

def main(svs_folder, tile_folder, output_file):
    """
    主函数，处理 SVS 文件与切片的匹配，过滤包含 'Warning' 的输出。
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = OutputFilter(original_stdout)
    sys.stderr = OutputFilter(original_stderr)

    try:
        if not os.path.exists(svs_folder):
            print(f"错误：SVS 文件夹 {svs_folder} 不存在")
            return
        if not os.path.exists(tile_folder):
            print(f"错误：切片文件夹 {tile_folder} 不存在")
            return

        print(f"开始处理 SVS 文件夹：{svs_folder}")
        print(f"切片文件夹：{tile_folder}")
        print(f"输出结果将保存到：{output_file}")
        match_svs_and_tiles(svs_folder, tile_folder, output_file)
        print(f"处理完成，结果已保存到 {output_file}")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

if __name__ == "__main__":
    svs_folder = r"F:\dataset\The Fourth Batch of Whole Slide Images\batch_six"
    tile_folder = r"G:\Output2"
    output_file = r"G:\svs_tile_positions_he_batch36.csv"
    main(svs_folder, tile_folder, output_file)