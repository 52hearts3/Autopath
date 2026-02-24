import os
import json
import pandas as pd
import gc
from tqdm import tqdm

def process_json_files(folder_path, output_csv, batch_size=1000):
    # 清空旧的输出文件
    if os.path.exists(output_csv):
        os.remove(output_csv)
        print(f"已删除旧文件 {output_csv}")

    json_files = []

    def collect_json_files(current_path):
        for entry in os.listdir(current_path):
            full_path = os.path.join(current_path, entry)
            if os.path.isdir(full_path):
                collect_json_files(full_path)
            elif full_path.endswith('.json'):
                json_files.append(full_path)

    collect_json_files(folder_path)
    print(f"找到 {len(json_files)} 个 JSON 文件需要处理。")

    first_batch = True
    contour_data = []
    global_index = 1  # 用于生成连续的全局索引

    for file_idx, file_path in enumerate(tqdm(json_files, desc="处理 JSON 文件"), 1):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            filename = os.path.basename(file_path)  # 获取文件名

            if 'nuc' in data:
                for nuc_id, nuc_data in data['nuc'].items():
                    if 'contour' in nuc_data and 'centroid' in nuc_data and 'type' in nuc_data:
                        contour_data.append({
                            'index': global_index,
                            'contour': str(nuc_data['contour']),
                            'centroid_x': nuc_data['centroid'][0],
                            'centroid_y': nuc_data['centroid'][1],
                            'type': nuc_data['type'],
                            'file_index': file_idx,
                            'filename': filename
                        })
                        global_index += 1

            if len(contour_data) >= batch_size:
                df = pd.DataFrame(contour_data,
                                columns=['index', 'contour', 'centroid_x', 'centroid_y', 'type', 'file_index', 'filename'])
                df = df.sort_values(by=['centroid_x', 'centroid_y'], ascending=[True, True])
                df[['index', 'contour', 'centroid_x', 'centroid_y', 'type', 'file_index', 'filename']].to_csv(
                    output_csv, mode='a', index=False, header=first_batch)
                first_batch = False
                contour_data = []
                gc.collect()

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")

    if contour_data:
        df = pd.DataFrame(contour_data,
                         columns=['index', 'contour', 'centroid_x', 'centroid_y', 'type', 'file_index', 'filename']) # index全局索引,file_index每个 JSON 文件分配从1开始的索引
        df = df.sort_values(by=['centroid_x', 'centroid_y'], ascending=[True, True])
        df[['index', 'contour', 'centroid_x', 'centroid_y', 'type', 'file_index', 'filename']].to_csv(
            output_csv, mode='a', index=False, header=first_batch)
        gc.collect()

    print(f"CSV 文件已保存为 {output_csv}")

if __name__ == "__main__":
    # 指定文件夹路径和输出 CSV 文件名
    folder_path = r"/media/lenovo/A06B2FA1620B6FCB/CT生成HE/dataset/json5"
    output_csv = r"/media/lenovo/A06B2FA1620B6FCB/CT生成HE/仁济/data-process/contours_output_group_batch51.csv"
    process_json_files(folder_path, output_csv, batch_size=10000)