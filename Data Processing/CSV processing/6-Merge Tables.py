import pandas as pd

# 文件路径
file1_path = r"G:\csv\output_with_numbering-B.csv"  # 第一个CSV文件路径
file2_path = r"G:\csv\output_with_numbering-B-batch3(1).csv"  # 第二个CSV文件路径
output_file = r"G:\csv\merged_output_batch34.csv"  # 输出合并后的CSV文件路径

# 读取两个CSV文件
try:
    df1 = pd.read_csv(file1_path, encoding='utf-8')
    print(f"成功读取第一个CSV文件，行数：{len(df1)}")
    df2 = pd.read_csv(file2_path, encoding='utf-8')
    print(f"成功读取第二个CSV文件，行数：{len(df2)}")
except Exception as e:
    print(f"读取CSV文件失败：{e}")
    exit()

# 检查两个文件的列是否相同
if list(df1.columns) != list(df2.columns):
    print("错误：两个CSV文件的列名不一致")
    print(f"第一个文件的列：{list(df1.columns)}")
    print(f"第二个文件的列：{list(df2.columns)}")
    exit()

# 按列合并（逐行拼接）
merged_df = pd.concat([df1, df2], axis=0, ignore_index=True)

# 打印合并后的信息
print(f"合并后的DataFrame行数：{len(merged_df)}")
print("合并后的列顺序：", merged_df.columns.tolist())
print(merged_df.head())

# 保存合并后的结果到新CSV文件
try:
    merged_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"已生成合并后的文件：{output_file}")
except Exception as e:
    print(f"保存合并后的CSV文件失败：{e}")