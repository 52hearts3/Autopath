# -*- coding: utf-8 -*-
"""
图像增强脚本（修复版）
修复问题：扫描到 0 个 JSON 文件
新增功能：
1. 详细调试日志
2. 路径存在性检查
3. 目录遍历追踪
4. 防止静默失败
"""

import os
import hashlib
from PIL import Image
import torch
from torchvision import transforms
import pandas as pd


# ========================================
# 1. 读取 txt 文件 → 按组返回路径列表
# ========================================
def read_grouped_paths(txt_path: str):
    """返回 [[group1_path1, ...], [group2_path1, ...], ...]"""
    groups = []
    current_group = []

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"配置文件不存在: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if set(line) <= {"-"} or line.startswith("---"):
                if current_group:
                    groups.append(current_group)
                    current_group = []
                continue
            # 去除首尾双引号
            path = line.strip('"')
            if not path:
                print(f"[WARN] 第 {line_num} 行路径为空，已跳过")
                continue
            current_group.append(path)

    if current_group:
        groups.append(current_group)

    print(f"从 {txt_path} 读取到 {len(groups)} 组路径")
    return groups


# ========================================
# 2. 递归读取所有 JSON 文件（带调试）
# ========================================
def load_json_files(folder_path: str):
    """递归遍历 folder_path（含子目录）返回所有 .json 文件的完整路径"""
    json_files = []

    if not os.path.exists(folder_path):
        print(f"   [ERROR] JSON 目录不存在: {folder_path}")
        return json_files
    if not os.path.isdir(folder_path):
        print(f"   [ERROR] 路径不是目录: {folder_path}")
        return json_files

    print(f"   [DEBUG] 开始扫描 JSON 目录: {folder_path}")
    try:
        file_count = 0
        for root, dirs, files in os.walk(folder_path):
            print(f"     [TRACE] 进入子目录: {root}")
            for file in files:
                if file.lower().endswith('.json'):
                    full_path = os.path.join(root, file)
                    json_files.append(full_path)
                    file_count += 1
                    if file_count <= 3:
                        print(f"       发现 JSON: {full_path}")
                    elif file_count == 4:
                        print(f"       ... 还有更多（共 {len(files)} 个文件在此目录）")
        print(f"   [INFO] 扫描完成，共发现 {len(json_files)} 个 .json 文件")
    except PermissionError as e:
        print(f"   [ERROR] 权限不足: {folder_path} → {e}")
    except Exception as e:
        print(f"   [ERROR] 遍历目录失败: {folder_path} → {e}")

    return json_files


# ========================================
# 3. 检查输入目录是否有图像
# ========================================
def check_input_images_exist(input_dir: str):
    required_names = [
        '右上 (Top-Right).png',
        '右下 (Bottom-Right).png',
        '左上 (Top-Left).png',
        '左下 (Bottom-Left).png'
    ]
    existing = []
    for name in required_names:
        path = os.path.join(input_dir, name)
        if os.path.exists(path):
            existing.append(name)
    return len(existing) > 0, existing


# ========================================
# 4. 图像增强核心函数
# ========================================
def augment_image(input_image_dir, output_dir, json_files, name_to_position):
    os.makedirs(output_dir, exist_ok=True)

    position_to_image = {
        '右上 (Top-Right)': os.path.join(input_image_dir, '右上 (Top-Right).png'),
        '右下 (Bottom-Right)': os.path.join(input_image_dir, '右下 (Bottom-Right).png'),
        '左上 (Top-Left)': os.path.join(input_image_dir, '左上 (Top-Left).png'),
        '左下 (Bottom-Left)': os.path.join(input_image_dir, '左下 (Bottom-Left).png')
    }

    processed = 0
    for json_file in json_files:
        base_name = os.path.splitext(os.path.basename(json_file))[0]

        if base_name not in name_to_position:
            print(f"   [SKIP] {base_name} → CSV 中无对应位置记录")
            continue

        position = name_to_position[base_name]
        img_path = position_to_image.get(position)

        if not img_path or not os.path.exists(img_path):
            print(f"   [SKIP] {base_name} → 图像不存在: {position} → {img_path}")
            continue

        try:
            image = Image.open(img_path).convert('L')

            transform = transforms.Compose([
                transforms.RandomRotation(degrees=(-90, 90)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomResizedCrop(size=image.size, scale=(0.8, 1.0)),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
                transforms.Resize((512, 512)),
            ])

            seed = int(hashlib.md5(base_name.encode()).hexdigest(), 16) % (2**32)
            torch.manual_seed(seed)

            aug = transform(image).convert('RGB')
            out_path = os.path.join(output_dir, f"{base_name}.png")
            aug.save(out_path)
            print(f"   [DONE] → {out_path}")
            processed += 1
        except Exception as e:
            print(f"   [ERROR] 处理 {base_name} 时出错: {e}")

    return processed


# ========================================
# 5. 主函数（增强版）
# ========================================
def main():
    # ================== 配置区 ==================
    INPUT_TXT   = r"I:\csv\第六批part2\T1增强\T1增强_input.txt"
    JSON_TXT    = r"I:\csv\第六批part2\ADC\ADC_json.txt"
    OUTPUT_TXT  = r"I:\csv\第六批part2\T1增强\T1增强_output.txt"
    CSV_PATH    = r"I:\csv\svs_tile_positions_he_batch_all.csv"
    # ===========================================

    print("="*80)
    print("图像增强工具启动（修复版）")
    print("="*80)

    # 检查配置文件
    for txt in [INPUT_TXT, JSON_TXT, OUTPUT_TXT, CSV_PATH]:
        if not os.path.exists(txt):
            raise FileNotFoundError(f"配置文件不存在: {txt}")

    print("正在读取 txt 文件...")
    input_groups  = read_grouped_paths(INPUT_TXT)
    json_groups   = read_grouped_paths(JSON_TXT)
    output_groups = read_grouped_paths(OUTPUT_TXT)

    print(f"共检测到 {len(input_groups)} 组任务")

    # 检查组数一致性
    for i, (in_g, js_g, out_g) in enumerate(zip(input_groups, json_groups, output_groups), 1):
        if not (len(in_g) == len(js_g) == len(out_g)):
            raise ValueError(
                f"第 {i} 组行数不匹配！\n"
                f"  input : {len(in_g)}\n"
                f"  json  : {len(js_g)}\n"
                f"  output: {len(out_g)}"
            )

    # 加载 CSV
    print(f"正在加载 CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    name_to_position = {
        row["Tile Filename"].replace('.png', ''): row["Position in HE Region"]
        for _, row in df.iterrows()
    }
    print(f"CSV 加载完成，共 {len(name_to_position)} 条记录")

    # ========================================
    # 第一阶段：预览 + 检查
    # ========================================
    print("\n" + "="*80)
    print("【预览】即将处理的图像增强任务")
    print("="*80)

    total_tasks = 0
    valid_tasks = 0
    error_dirs = []

    for grp_idx, (in_paths, js_paths, out_paths) in enumerate(zip(input_groups, json_groups, output_groups), 1):
        print(f"\n第 {grp_idx} 组（共 {len(in_paths)} 个子任务）")
        for idx, (in_dir, js_dir, out_dir) in enumerate(zip(in_paths, js_paths, out_paths), 1):
            # 检查输入图像
            has_image, existing_imgs = check_input_images_exist(in_dir)
            json_files = load_json_files(js_dir)
            json_count = len(json_files)

            total_tasks += json_count
            if has_image and json_count > 0:
                valid_tasks += json_count

            status = "有效" if (has_image and json_count > 0) else "无效"
            print(f"  [{idx}] {status}")
            print(f"     输入图像 : {in_dir}")
            print(f"     JSON 目录 : {js_dir}")
            print(f"     输出目录 : {out_dir}")
            print(f"     JSON 数量 : {json_count} 个")
            print(f"     图像状态 : {'有' if has_image else '无'} "
                  f"({', '.join(existing_imgs) if existing_imgs else '无'})")

            if not has_image:
                error_dirs.append(in_dir)

    print(f"\n" + "-"*80)
    print(f"总计扫描到 {total_tasks} 个 JSON 文件")
    print(f"其中有效任务（有图像 + 有 JSON）: {valid_tasks} 个")
    if error_dirs:
        print(f"警告：{len(error_dirs)} 个输入目录缺少必需图像")
    print("-"*80)

    # ========================================
    # 第二阶段：用户确认
    # ========================================
    if error_dirs:
        confirm = input("\n检测到缺失图像，是否继续？(y/N): ")
        if confirm.lower() not in ['y', 'yes', '是']:
            print("操作已取消。")
            return

    input("\n按回车键开始写入图像...（或 Ctrl+C 取消）\n")

    # ========================================
    # 第三阶段：执行增强
    # ========================================
    print("\n开始图像增强与保存...\n")
    total_processed = 0

    for grp_idx, (in_paths, js_paths, out_paths) in enumerate(zip(input_groups, json_groups, output_groups), 1):
        print(f"\n{'=' * 60}")
        print(f"处理第 {grp_idx} 组（共 {len(in_paths)} 个）")
        print(f"{'=' * 60}")

        for idx, (in_dir, js_dir, out_dir) in enumerate(zip(in_paths, js_paths, out_paths), 1):
            print(f"\n→ 子任务 {idx}/{len(in_paths)}")
            print(f"   输入: {in_dir}")
            print(f"   JSON : {js_dir}")
            print(f"   输出: {out_dir}")

            # 再次检查
            if not os.path.exists(in_dir):
                print(f"   [ERROR] 输入目录不存在，跳过")
                continue
            if not os.path.exists(js_dir):
                print(f"   [ERROR] JSON 目录不存在，跳过")
                continue

            has_image, _ = check_input_images_exist(in_dir)
            if not has_image:
                print(f"   [SKIP] 无必需图像，跳过")
                continue

            json_files = load_json_files(js_dir)
            if not json_files:
                print(f"   [WARN] 无 JSON 文件，跳过")
                continue

            count = augment_image(in_dir, out_dir, json_files, name_to_position)
            total_processed += count

    print(f"\n{'='*60}")
    print(f"全部处理完成！共成功增强 {total_processed} 张图像。")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()