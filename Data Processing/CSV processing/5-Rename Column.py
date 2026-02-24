import pandas as pd

# 输入和输出文件路径
input_file = r"G:\output_with_numbering-B-batch52.csv"      # 请替换为你的实际路径
output_file = r"G:\csv\output_with_numbering-B-batch52.csv"      # 修改后的文件保存路径

# 读取 CSV 文件
try:
    df = pd.read_csv(input_file, encoding='utf-8')
    print(f"成功读取CSV文件，列名：{df.columns.tolist()}")
except Exception as e:
    print(f"读取CSV文件失败：{e}")
    exit()

# 检查并修改列名
if 'index_range' in df.columns:
    df.rename(columns={'index_range': '检查结论'}, inplace=True)
    print("已将列名 'index_range' 修改为 '检查结论'")
else:
    print("错误：未找到列名 'index_range'")
    exit()

# 保存修改后的 CSV 文件
try:
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"已保存修改后的文件：{output_file}")
except Exception as e:
    print(f"保存CSV文件失败：{e}")
