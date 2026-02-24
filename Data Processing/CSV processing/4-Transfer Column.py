import pandas as pd

# 文件路径
input_file = r"F:\grouped_index_ranges_batch51.csv" # 输入CSV文件路径
output_file = r"F:\grouped_index_ranges_batch51.csv" # 输出CSV文件路径

# 读取CSV文件
try:
    df = pd.read_csv(input_file, encoding='utf-8')
    print(f"成功读取CSV文件，行数：{len(df)}")
except Exception as e:
    print(f"读取CSV文件失败：{e}")
    exit()

# 检查'编号'列是否存在
if '编号' not in df.columns:
    print("错误：CSV文件中没有'编号'列")
    exit()

# 将'编号'列移动到最前面
cols = ['编号'] + [col for col in df.columns if col != '编号']
df = df[cols]

# 打印前几行以确认
print("调整后的列顺序：", df.columns.tolist())
print(df.head())

# 保存到新的CSV文件
try:
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"已生成新文件：{output_file}")
except Exception as e:
    print(f"保存CSV文件失败：{e}")