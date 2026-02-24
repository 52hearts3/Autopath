import os
import json
import h5py
import uuid
import numpy as np
from tqdm import tqdm


def json_to_h5py(input_dir, output_h5_file):
    """
    读取目录中的 JSON 文件并将其追加存储到 HDF5 文件中，使用差值法提高压缩率。
    - 使用文件名（不含路径）作为 HDF5 根组名称（dataname）。
    - 'mag' 作为属性存储。
    - 每个 JSON 文件的核数据存储在一个组内，'contours'（变长扁平数组，首个坐标对 int16，后续差值 int8，gzip 压缩），'types'（int8，无压缩）作为数据集，核 ID 存储为属性。
    - contours 的第一对坐标保持不变（int16），后续坐标存储与前一对的差值（int8）。
    - 不存储 bbox、centroid 和 type_prob。
    - 添加进度条显示处理进度。
    - 验证 contour 长度为偶数（坐标对），否则跳过该核。
    - 如果组名已存在于 HDF5 文件中，跳过该 JSON 文件的处理。

    参数：
        input_dir (str): 包含 JSON 文件的目录路径
        output_h5_file (str): 输出 HDF5 文件的路径
    """

    # 收集所有 JSON 文件路径
    def collect_json_files(directory):
        json_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
        return json_files

    # 获取 JSON 文件列表
    json_files = collect_json_files(input_dir)
    print(f"找到 {len(json_files)} 个 JSON 文件")

    # 以追加模式打开 HDF5 文件（'a' 模式支持读写和追加）
    with h5py.File(output_h5_file, 'a') as h5f:
        # 使用进度条处理 JSON 文件
        for item_path in tqdm(json_files, desc="处理 JSON 文件", unit="file"):
            try:
                # 使用文件名（不含 .json 扩展名）作为组名称
                group_name = os.path.splitext(os.path.basename(item_path))[0]

                # 检查组名称是否已存在，若存在则跳过
                if group_name in h5f:
                    print(f"组 {group_name} 已存在于 HDF5 文件中，跳过处理 {item_path}")
                    continue

                # 读取 JSON 文件
                with open(item_path, 'r') as f:
                    json_data = json.load(f)

                # 创建主组
                main_grp = h5f.create_group(group_name)
                print(f"已创建组 {group_name} 于 HDF5")

                # 存储 'mag' 作为属性
                main_grp.attrs['mag'] = str(json_data.get('mag', 'None'))

                # 收集核数据（过滤无效 contour）
                contours = []
                types = []
                nuc_ids = []
                skipped_nucs = 0
                for nuc_id, nuc_data in json_data.get('nuc', {}).items():
                    try:
                        contour_list = nuc_data['contour']
                        # 转换为 NumPy 数组并重塑为 (N, 2)
                        contour_array = np.array(contour_list, dtype=np.int16).reshape(-1, 2)

                        # 验证：至少有一个坐标对
                        if len(contour_array) == 0:
                            print(f"警告：{item_path} 中的核 {nuc_id} 的 contour 为空，跳过该核")
                            skipped_nucs += 1
                            continue

                        # 使用差值法：第一对坐标不变（int16），后续坐标存储差值（int8）
                        diff_contour = np.zeros_like(contour_array, dtype=np.int16)
                        diff_contour[0] = contour_array[0]  # 保留第一个坐标对（int16）
                        if len(contour_array) > 1:
                            # 计算差值
                            diff_values = contour_array[1:] - contour_array[:-1]
                            # 检查差值是否在 int8 范围内 (-128 到 127)
                            if np.any(diff_values < -128) or np.any(diff_values > 127):
                                print(f"警告：{item_path} 中的核 {nuc_id} 的差值超出 int8 范围，跳过该核")
                                skipped_nucs += 1
                                continue
                            diff_contour[1:] = diff_values.astype(np.int8)

                        # 扁平化为1D数组
                        diff_contour_flat = diff_contour.flatten()

                        contours.append(diff_contour_flat)
                        types.append(nuc_data['type'])
                        nuc_ids.append(nuc_id)
                    except Exception as nuc_e:
                        print(f"警告：处理 {item_path} 中的核 {nuc_id} 时出错：{str(nuc_e)}，跳过该核")
                        skipped_nucs += 1
                        continue

                if skipped_nucs > 0:
                    print(f"注意：{item_path} 中跳过了 {skipped_nucs} 个无效核")

                # 存储 contours（使用变长数组，扁平差值坐标）
                if contours:  # 仅在 contours 非空时创建数据集
                    # 定义变长数据类型（一维 int16，兼容 int8 差值）
                    vlen_dtype = h5py.vlen_dtype(np.dtype('int16'))
                    # 创建数据集，启用 gzip 压缩
                    contours_ds = main_grp.create_dataset(
                        'contours',
                        shape=(len(contours),),
                        dtype=vlen_dtype,
                        compression='gzip',
                        compression_opts=9
                    )
                    # 逐个存储变长数组
                    for i, contour in enumerate(contours):
                        contours_ds[i] = contour
                else:
                    # 如果 contours 为空，创建一个空数据集
                    main_grp.create_dataset('contours', shape=(0,), dtype=h5py.vlen_dtype(np.dtype('int16')))

                # 存储 types
                types_array = np.array(types, dtype=np.int8)
                main_grp.create_dataset('types', data=types_array)

                # 存储核 ID 作为属性
                main_grp.attrs['nuc_ids'] = nuc_ids

                print(f"已存储 {item_path} 到 HDF5，组名称为 {group_name}")

            except Exception as e:
                print(f"处理 {item_path} 时出错：{str(e)}")
                continue  # 继续处理下一个文件，避免中断


def read_h5py_nuc(h5_file, group_name):
    """
    从 HDF5 文件中读取指定组的所有核数据，并解码差值坐标（首个坐标 int16，后续差值 int8）。

    参数：
        h5_file (str): HDF5 文件的路径
        group_name (str): 主组名称（文件名，不含 .json）

    返回：
        dict: 返回核数据的字典，键为 nuc_id，值为 {'contour': ..., 'type': ...}；若未找到则返回 None
    """
    try:
        # 确保 group_name 是字符串
        if isinstance(group_name, tuple):
            print(f"警告：group_name 是元组 {group_name}，尝试提取第一个元素")
            group_name = group_name[0] if group_name else ""
        if not isinstance(group_name, (str, bytes)):
            raise TypeError(f"无效的 group_name 类型：{type(group_name)}，需要 str 或 bytes")

        with h5py.File(h5_file, 'r') as h5f:
            main_grp = h5f.get(group_name)
            if main_grp is None:
                print(f"在 {h5_file} 中未找到组 {group_name}")
                print("可用组：")
                h5f.visit(lambda name: print(f"  {name}") if isinstance(h5f[name], h5py.Group) else None)
                return None

            data = {}
            contours = main_grp['contours'][()]
            types = main_grp['types'][()]
            nuc_ids = main_grp.attrs.get('nuc_ids', [])

            for i, nuc_id in enumerate(nuc_ids):
                # 获取扁平的差值坐标
                contour_flat = contours[i]
                if len(contour_flat) % 2 != 0:
                    print(f"警告：读取核 {nuc_id} 时，contour 长度 {len(contour_flat)} 为奇数，无法重塑")
                    continue
                # 重塑为 (N, 2)
                diff_contour = contour_flat.reshape(-1, 2).astype(np.int16)  # 确保 int16 处理
                # 解码差值：累加还原原始坐标
                original_contour = np.zeros_like(diff_contour, dtype=np.int16)
                original_contour[0] = diff_contour[0]  # 第一个坐标对不变（int16）
                if len(diff_contour) > 1:
                    for j in range(1, len(diff_contour)):
                        original_contour[j] = original_contour[j-1] + diff_contour[j].astype(np.int16)
                # 转换为列表
                contour_2d = original_contour.tolist()
                data[nuc_id] = {
                    'contour': contour_2d,
                    'type': int(types[i])
                }
            return data
    except Exception as e:
        print(f"读取组 {group_name} 时出错：{str(e)}")
        return None


def list_h5py_datasets(h5_file):
    """
    列出 HDF5 文件中的所有组和数据集

    参数：
        h5_file (str): HDF5 文件的路径
    """
    def print_items(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"数据集: {name}")
        elif isinstance(obj, h5py.Group):
            print(f"组: {name}")

    try:
        with h5py.File(h5_file, 'r') as h5f:
            print(f"HDF5 文件内容: {h5_file}")
            h5f.visititems(print_items)
    except Exception as e:
        print(f"读取 HDF5 文件 {h5_file} 时出错：{str(e)}")


# 示例用法
if __name__ == "__main__":
    # 指定输入目录和输出 HDF5 文件
    input_directory = r"I:\json5-压缩"  # 替换为您的目录路径
    output_hdf5_file = r"H:\output.h5" # 替换为所需的输出文件路径

    # 将 JSON 文件转换为 HDF5
    # json_to_h5py(input_directory, output_hdf5_file)

    # 列出 HDF5 文件中的所有组和数据集
    # list_h5py_datasets(output_hdf5_file)

    # 示例：读取特定组的所有核数据
    group_name = '24-15899-K-2025-06-02_22_29_57_tile_8192_69632'
    data = read_h5py_nuc(output_hdf5_file, group_name)
    if data:
        print("检索到的核数据:", data)