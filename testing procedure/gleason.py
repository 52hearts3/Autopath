import pandas as pd
from collections import Counter
import re

# ==================== 配置路径 ====================
match_file = r"G:\分析测试\匹配txt\10007\generated_numbers_all.txt"
pathology_file = r"G:\病理报告合并.xlsx"

GLEASON_TOP_K = 4 # 格里森评分显示前几种

# ==================== 1. 提取所有出现过的患者前缀 ====================
with open(match_file, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

prefixes = []
for line in lines:
    prefix = ''
    for c in line:
        if c.isalpha():
            prefix += c
        else:
            break
    if prefix:
        prefixes.append(prefix)

# 去重 + 计数（但这里我们只关心有哪些前缀存在）
unique_prefixes = list(set(prefixes))  # 所有出现过的不同前缀
total_patients = len(prefixes)         # 总匹配次数（可能有重复患者）
total_unique_prefixes = len(unique_prefixes)

print(f"共提取到 {total_unique_prefixes} 种不同前缀，"
      f"总共匹配到 {total_patients} 条记录（可能含重复患者）")
print(f"涉及的前缀：{', '.join(sorted(unique_prefixes))}")
print("-" * 80)

# ==================== 2. 读取病理报告并筛选所有匹配前缀的患者 ====================
df = pd.read_excel(pathology_file, sheet_name='Sheet1')
df.columns = ['编号', '格里森', '神经侵犯', '脉管内癌栓', '浸润']

# 提取编号前缀
df['前缀'] = df['编号'].astype(str).str.extract(r'^([A-Z]+)', expand=False)

# 筛选出所有匹配到的前缀患者
high_freq_df = df[df['前缀'].isin(unique_prefixes)].copy()

if high_freq_df.empty:
    raise ValueError("在病理报告中未找到任何匹配前缀的患者！")

print(f"筛选后有效患者数量：{len(high_freq_df)} 人")
print("-" * 80)

# ==================== 3. 格里森评分：统计出现频率，列出前几种 ====================
gleason_counts = high_freq_df['格里森'].value_counts()

# 过滤掉 NaN 和 '不适用' 等无效值（可根据实际数据增加其他过滤条件）
invalid_values = [float('nan'), '不适用', '无', None, '']
gleason_counts = gleason_counts[~gleason_counts.index.isin(invalid_values)]

total_valid = gleason_counts.sum()
top_gleason = gleason_counts.head(GLEASON_TOP_K)

# 格式化输出
gleason_lines = []
for gleason, count in top_gleason.items():
    percent = count / total_valid * 100 if total_valid > 0 else 0
    gleason_lines.append(f"{gleason}（{count}人，{percent:.1f}%）")

final_gleason_conclusion = "； ".join(gleason_lines) if gleason_lines else "无有效格里森评分数据"

# ==================== 4. 其他三项：取出现最多的（众数） ====================
final_nerve = high_freq_df['神经侵犯'].mode().iloc[0] if not high_freq_df['神经侵犯'].mode().empty else '未知'
final_vessel = high_freq_df['脉管内癌栓'].mode().iloc[0] if not high_freq_df['脉管内癌栓'].mode().empty else '未知'
final_invasion = high_freq_df['浸润'].mode().iloc[0] if not high_freq_df['浸润'].mode().empty else '未知'

# ==================== 5. 输出最终报告 ====================
print("【最终综合病理报告 - 全部匹配患者统计】".center(70, "="))
print(f"患者群体：所有匹配编号前缀的患者（共 {len(high_freq_df)} 人）")
print(f"涉及前缀数量：{total_unique_prefixes} 种")
print(f"统计方法：")
print(f"  • 格里森评分 → 列出出现频率最高的前 {GLEASON_TOP_K} 种（含频次与占比）")
print(f"  • 其他项目 → 取出现次数最多的结果（众数）")
print("-" * 80)
print(f"{'项目':<12} {'结论':<50} {'选取原则'}")
print("-" * 80)
print(f"{'格里森评分':<12} {final_gleason_conclusion:<50} {'出现频率最高的前{GLEASON_TOP_K}种'}")
print(f"{'神经侵犯':<12} {final_nerve:<50} {'出现最多'}")
print(f"{'脉管内癌栓':<12} {final_vessel:<50} {'出现最多'}")
print(f"{'浸润':<12} {final_invasion:<50} {'出现最多'}")
print("=" * 80)