import pandas as pd
from tqdm import tqdm
import os
import csv

def sort_large_csv(input_csv, output_csv, chunk_size=100000):
    """
    对大型CSV文件进行整体排序，按centroid_x, centroid_y, type排序，仅保留四列的行，
    并将index列替换为排序后的全局索引。

    Parameters:
    input_csv (str): 输入CSV文件路径
    output_csv (str): 输出排序后CSV文件路径
    chunk_size (int): 每次读取的行数
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_csv):
        print(f"错误：输入文件 {input_csv} 不存在")
        return

    # 获取CSV文件的总行数以显示进度
    total_rows = sum(1 for _ in open(input_csv)) - 1  # 减去表头行
    print(f"总行数: {total_rows}")

    # 初始化一个空的DataFrame列表，用于存储分块数据
    all_chunks = []

    # 预期列
    expected_columns = ['index', 'contour', 'centroid_x', 'centroid_y', 'type', 'file_index', 'filename']

    # 分块读取CSV文件，显式指定列名和类型
    try:
        for chunk in tqdm(
            pd.read_csv(
                input_csv,
                chunksize=chunk_size,
                quoting=csv.QUOTE_NONNUMERIC,
                usecols=expected_columns,
                dtype={'contour': str, 'type': str, 'centroid_x': float, 'centroid_y': float}
            ),
            total=total_rows // chunk_size + 1,
            desc="读取CSV分块"
        ):
            # 过滤掉缺失centroid_x或centroid_y的行
            valid_rows = chunk.dropna(subset=['centroid_x', 'centroid_y'])
            if not valid_rows.empty:
                all_chunks.append(valid_rows)
    except pd.errors.ParserError as e:
        print(f"解析CSV文件时出错: {str(e)}")
        print("请检查输入CSV文件格式，确保所有行字段数一致")
        return
    except ValueError as e:
        print(f"读取CSV文件时出错: {str(e)}")
        print("可能的原因：CSV文件中存在非预期的列或格式错误")
        return

    # 检查是否有有效数据
    if not all_chunks:
        print("错误：没有找到包含四列的有效数据")
        return

    # 合并所有分块数据
    print("合并所有分块数据...")
    df = pd.concat(all_chunks, ignore_index=True)

    # 按centroid_x, centroid_y, type进行整体排序
    print("执行整体排序...")
    df_sorted = df.sort_values(by=['centroid_x', 'centroid_y', 'type'], ascending=[True, True, True])

    # 替换index列为排序后的全局索引（从1开始）
    print("生成全局索引...")
    df_sorted['index'] = range(1, len(df_sorted) + 1)

    # 保存排序后的数据到新的CSV文件，包含所有列
    print(f"保存排序结果到 {output_csv}...")
    df_sorted.to_csv(
        output_csv, index=False, quoting=csv.QUOTE_NONNUMERIC
    )
    print(f"排序完成，文件已保存为 {output_csv}")


if __name__ == "__main__":
    # 指定输入和输出文件路径
    input_csv = r"/media/lenovo/A06B2FA1620B6FCB/CT生成HE/仁济/data-process/contours_output_group_batch51.csv"
    output_csv = r"/media/lenovo/A06B2FA1620B6FCB/CT生成HE/仁济/data-process/data/contours_sorted_output_group_batch51.csv"
    sort_large_csv(input_csv, output_csv, chunk_size=100000)