import csv
from pathlib import Path
import os


def match_filenames_and_save(
        txt_path: str,
        csv_path: str,
        output_txt_path: str = "matched_filenames.txt"
) -> None:
    """
    根据 txt 中的编号，从 csv 中查找对应的文件名，去掉 .json 后缀后保存到新的 txt
    """
    # 1. 读取 txt 中的所有编号（保持顺序，去除空行和空白）
    numbers = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            num = line.strip()
            if num:
                numbers.append(num)

    print(f"从 txt 中读取到 {len(numbers)} 个编号")

    # 2. 建立 编号 → 文件名 的字典
    number_to_filename = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        # 如果 CSV 有标题行，取消下面注释
        # next(reader)

        for row in reader:
            if len(row) >= 2:
                num = row[0].strip()
                filename = row[1].strip()
                if num:
                    number_to_filename[num] = filename

    print(f"从 CSV 中读取到 {len(number_to_filename)} 条映射记录")

    # 3. 按 txt 顺序查找，并去掉 .json 后缀
    matched_results = []
    not_found_count = 0

    for num in numbers:
        filename = number_to_filename.get(num)
        if filename is not None:
            # 去掉 .json 后缀（不区分大小写）
            name, ext = os.path.splitext(filename)
            if ext.lower() == '.json':
                cleaned = name
            else:
                cleaned = filename  # 没有 .json 就保持原样
            matched_results.append(cleaned)
        else:
            matched_results.append(f"{num}  ← 未找到")
            not_found_count += 1

    # 4. 写入结果
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for item in matched_results:
            f.write(item + '\n')

    print(f"已完成匹配，已保存到：{output_txt_path}")
    print(f"总编号数量：{len(numbers)}")
    print(f"成功匹配：{len(matched_results) - not_found_count}")
    print(f"未找到：{not_found_count}")


# ─────────────── 使用示例 ───────────────
if __name__ == "__main__":
    txt_file   = r'G:\分析测试\匹配txt\verify\X24-4156\L4000\extracted_numbers_part1.txt'
    csv_file   = r"G:\分析测试\test.csv"
    output_file = r"G:\分析测试\匹配txt\verify\X24-4156\L4000\result1.txt"

    match_filenames_and_save(txt_file, csv_file, output_file)