import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# ====================== 硬编码的配置区域 ======================
H5_FILE_PATH = r"G:\output.h5"  # HDF5 文件路径
TXT_FILE_PATH = r"G:\分析测试\匹配txt\10005\8\matched_result1.txt"  # 包含 tile 名称的 txt 文件
SHOW_MASK =True  # 是否逐个弹出 matplotlib 窗口显示
SAVE_MASK = False  # 是否保存掩码图片
SAVE_DIR = "masks_output_2025"  # 保存文件夹（会自动创建）
RESIZE_TO = 512  # 输出/保存的图像尺寸（像素）


# =============================================================


def read_group_names_from_txt(txt_path):
    """读取 txt 文件中的所有 group/tile 名称"""
    if not os.path.isfile(txt_path):
        print(f"错误：txt 文件不存在 → {txt_path}")
        return []

    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"从 {txt_path} 读取到 {len(lines)} 个 tile 名称")
    return lines


def read_h5py_nuc(h5_file, group_name):
    """从 HDF5 读取单个组的核数据并还原坐标"""
    try:
        with h5py.File(h5_file, 'r') as h5f:
            main_grp = h5f.get(group_name)
            if main_grp is None:
                print(f"  未找到组：{group_name}")
                return None

            data = {}
            contours = main_grp['contours'][()]
            types = main_grp['types'][()]
            nuc_ids = main_grp.attrs.get('nuc_ids', [])

            for i, nuc_id in enumerate(nuc_ids):
                contour_flat = contours[i]
                if len(contour_flat) % 2 != 0:
                    print(f"  警告：核 {nuc_id} contour 长度奇数，跳过")
                    continue

                diff_contour = contour_flat.reshape(-1, 2).astype(np.int16)
                original_contour = np.zeros_like(diff_contour, dtype=np.int16)
                original_contour[0] = diff_contour[0]

                if len(diff_contour) > 1:
                    cumsum = np.cumsum(diff_contour[1:].astype(np.int16), axis=0)
                    original_contour[1:] = original_contour[0] + cumsum

                data[nuc_id] = {
                    'contour': original_contour.tolist(),
                    'type': int(types[i])
                }
            return data
    except Exception as e:
        print(f"  读取 {group_name} 失败：{str(e)}")
        return None


def generate_colored_mask(nuclei, size=2048):
    """生成 2048×2048 的彩色掩码"""
    mask = np.zeros((size, size, 3), dtype=np.uint8)

    # BGR 颜色
    type_colors = {
        0: (255, 255, 255),  # nolabe - white
        1: (255, 0, 0),  # neopla - blue
        2: (0, 255, 0),  # inflam - green
        3: (0, 0, 255),  # connec - red
        4: (0, 255, 255),  # necros - yellow
        5: (0, 165, 255),  # no-neo - orange
    }

    for nuc_data in nuclei.values():
        contour = nuc_data.get('contour', [])
        nuc_type = nuc_data.get('type', 3)
        if not contour:
            continue

        pts = np.array(contour, dtype=np.int32)
        color = type_colors.get(nuc_type, (0, 0, 255))
        cv2.fillPoly(mask, [pts], color=color)

    return mask


def main():
    group_names = read_group_names_from_txt(TXT_FILE_PATH)
    if not group_names:
        print("没有读取到任何 tile 名称，程序退出。")
        return

    if SAVE_MASK:
        os.makedirs(SAVE_DIR, exist_ok=True)
        print(f"掩码图片将保存到：{os.path.abspath(SAVE_DIR)}")

    print(f"总共处理 {len(group_names)} 个 tile ...\n")

    for group_name in tqdm(group_names, desc="处理进度"):
        nuclei = read_h5py_nuc(H5_FILE_PATH, group_name)
        if not nuclei:
            continue

        # 生成完整掩码
        mask_full = generate_colored_mask(nuclei, size=2048)

        # 缩放
        mask_resized = cv2.resize(mask_full, (RESIZE_TO, RESIZE_TO), interpolation=cv2.INTER_AREA)

        # 显示（如果开启）
        if SHOW_MASK:
            mask_rgb = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(6, 6))
            plt.imshow(mask_rgb)
            plt.title(group_name)
            plt.axis('off')
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.4)
            plt.close()

        # 保存（如果开启）
        if SAVE_MASK:
            save_path = os.path.join(SAVE_DIR, f"{group_name}.png")
            cv2.imwrite(save_path, mask_resized)

    print("\n全部处理完成！")


if __name__ == "__main__":
    main()