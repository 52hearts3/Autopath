import pandas as pd
import os
from tqdm import tqdm
import time

def create_grouped_excel(input_csv, output_excel, chunk_size=1000000):
    """
    按file_index分组，生成Excel表格，包含filename和index_range列。
    使用Pandas分块读取CSV，合并后分组，显示带ETA的进度条。

    Parameters:
    input_csv (str): 输入CSV文件路径
    output_excel (str): 输出Excel文件路径
    chunk_size (int): 分块读取时每块的行数
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_csv):
        print(f"错误：输入文件 {input_csv} 不存在")
        return

    # 定义数据类型以优化内存
    dtypes = {
        'index': 'int32',
        'file_index': 'int32',
        'filename': 'category',
        'contour': str,
        'centroid_x': 'float32',
        'centroid_y': 'float32',
        'type': 'category'
    }

    # 获取CSV文件总行数以显示进度
    print("计算总行数...")
    total_rows = sum(1 for _ in open(input_csv)) - 1  # 减去表头行
    print(f"总行数: {total_rows}")

    # 分块读取CSV并合并
    print("分块读取CSV并合并...")
    all_chunks = []
    with tqdm(total=total_rows, desc="读取CSV分块", unit="rows", dynamic_ncols=True) as pbar:
        for chunk in pd.read_csv(
            input_csv,
            chunksize=chunk_size,
            dtype=dtypes,
            usecols=['index', 'file_index', 'filename']
        ):
            all_chunks.append(chunk)
            pbar.update(len(chunk))

    # 合并所有分块
    print("合并所有分块数据...")
    df = pd.concat(all_chunks, ignore_index=True)

    # 按file_index分组
    print("按file_index分组...")
    grouped = df.groupby('file_index')
    total_groups = len(grouped)  # 分组数量
    print(f"总分组数: {total_groups}")

    # 收集filename和index_range
    result_data = []
    with tqdm(total=total_groups, desc="处理分组", unit="groups", dynamic_ncols=True) as pbar:
        for file_index, group in grouped:
            # 获取filename（假设file_index与filename一对一）
            filename = group['filename'].iloc[0]
            # 按index排序并生成index_range
            indices = sorted(group['index'].values)
            index_range = '-'.join(map(str, indices)) if indices else ''
            result_data.append({'filename': filename, 'index_range': index_range})
            pbar.update(1)

    # 创建结果DataFrame
    result_df = pd.DataFrame(result_data, columns=['filename', 'index_range'])

    # 按filename排序
    result_df = result_df.sort_values(by='filename')

    # 保存到Excel
    print(f"保存结果到 {output_excel}...")
    start_time = time.time()
    result_df.to_excel(output_excel, index=False, engine='openpyxl')
    print(f"Excel保存耗时: {time.time() - start_time:.2f}秒")
    print(f"Excel文件已保存为 {output_excel}")


if __name__ == "__main__":
    # 指定输入和输出文件路径
    input_csv = r"/media/lenovo/A06B2FA1620B6FCB/CT生成HE/仁济/data-process/data/contours_sorted_output_group_batch51.csv"
    output_excel = r"/media/lenovo/A06B2FA1620B6FCB/CT生成HE/仁济/data-process/data/grouped_index_ranges_batch51.xlsx"
    create_grouped_excel(input_csv, output_excel, chunk_size=1000000)