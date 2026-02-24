import pandas as pd
import os

# 文件路径
input_file = r"F:\grouped_index_ranges_batch51.xlsx"
output_file = r"F:\grouped_index_ranges_batch51.csv"
folder_path = r"I:\json5"

# ========== 新增：指定起始前缀 ==========
start_prefix = "AACU"  # 你可以改成任意四位字母
# ======================================

# 读取CSV
try:
    df = pd.read_excel(input_file)
    print(f"成功读取CSV文件，行数：{len(df)}")
except Exception as e:
    print(f"读取CSV文件失败：{e}")
    exit()

# 检查filename列是否存在
if 'filename' not in df.columns:
    print("错误：CSV文件中没有'filename'列")
    exit()

# 将字母前缀转换为整数索引
def prefix_to_index(prefix: str) -> int:
    prefix = prefix.upper()
    index = 0
    for ch in prefix:
        index = index * 26 + (ord(ch) - 65)
    return index

# 将整数索引转换为四位字母前缀
def index_to_prefix(index: int) -> str:
    letters = ''
    temp_index = index
    for _ in range(4):
        letters = chr(65 + (temp_index % 26)) + letters
        temp_index //= 26
    return letters

# 获取文件夹与filename的映射（保持不变）
def get_folder_filename_mapping(folder_path):
    folder_to_files = {}
    if not os.path.exists(folder_path):
        print(f"错误：文件夹路径 {folder_path} 不存在")
        return folder_to_files
    for folder_name in os.listdir(folder_path):
        if folder_name.startswith('.'):
            print(f"跳过隐藏文件夹：{folder_name}")
            continue
        folder_full_path = os.path.join(folder_path, folder_name)
        if os.path.isdir(folder_full_path):
            folder_to_files[folder_name] = []
            def recursive_find_json(current_path):
                for item in os.listdir(current_path):
                    item_path = os.path.join(current_path, item)
                    if os.path.isdir(item_path):
                        recursive_find_json(item_path)
                    elif item.endswith('.json'):
                        folder_to_files[folder_name].append(item.replace('.json', ''))
            recursive_find_json(folder_full_path)
            print(f"文件夹 {folder_name} 找到 {len(folder_to_files[folder_name])} 个文件")
    return folder_to_files

# 生成唯一编号
def generate_unique_id(index, letter_prefix):
    num_base = 100000
    num_part = index % num_base
    num_str = str(num_part).zfill(5)
    return f"{letter_prefix}{num_str}"

# ========== 修改：从指定前缀开始 ==========
start_index = prefix_to_index(start_prefix)
# ========================================

# 获取文件夹与filename的映射
folder_to_files = get_folder_filename_mapping(folder_path)
if not folder_to_files:
    print("警告：没有找到任何编号文件夹或文件")
    exit()

df['编号'] = ''
folder_prefixes = {}
current_folder_index = 0

for folder_name, filenames in folder_to_files.items():
    if folder_name not in folder_prefixes:
        folder_prefixes[folder_name] = index_to_prefix(start_index + current_folder_index)
        print(f"文件夹 {folder_name} 分配前缀：{folder_prefixes[folder_name]}")
        current_folder_index += 1
    for idx, filename in enumerate(filenames):
        mask = df['filename'].str.replace('.json', '', regex=False) == filename
        if mask.any():
            new_id = generate_unique_id(idx, folder_prefixes[folder_name])
            df.loc[mask, '编号'] = new_id
            print(f"文件 {filename}.json 分配编号：{new_id}")
        else:
            print(f"警告：文件 {filename}.json 在CSV中未找到匹配")

unassigned_rows = df[df['编号'] == '']
if not unassigned_rows.empty:
    print(f"警告：{len(unassigned_rows)} 行未分配编号，示例filename：{unassigned_rows['filename'].head().tolist()}")

try:
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"已生成新文件：{output_file}")
except Exception as e:
    print(f"保存CSV文件失败：{e}")
