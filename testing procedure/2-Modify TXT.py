import csv

# 输入文件路径
input_file = r"G:\分析测试\匹配txt\verify\X24-4156\L4000\generated_numbers_2026_part4.txt"
# 输出文件路径（只包含编号，每行一个）
output_file = r'G:\分析测试\匹配txt\verify\X24-4156\L4000\extracted_numbers_part4.txt'

# 读取 TSV 文件，提取第三列（提取的编号）
with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)  # 跳过标题行
    numbers = [row[2] for row in reader if len(row) >= 3]

# 保存到新文件
with open(output_file, 'w', encoding='utf-8') as out:
    for num in numbers:
        out.write(num + '\n')

print(f"提取完成，共 {len(numbers)} 个编号，已保存到 {output_file}")