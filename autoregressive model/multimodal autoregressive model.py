import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import Counter
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from transformers import AutoModel, AutoTokenizer
from multiprocessing import Pool, cpu_count
import pickle
from numba import jit
import cv2
import re
import torch.cuda.amp as amp
from typing import Optional, Tuple


# 词汇表类
class Vocabulary:
    def __init__(self, min_freq=1):
        self.word2idx = {}
        self.idx2word = {}
        self.word2idx['<PAD>'] = 0
        self.word2idx['<BOS>'] = 1
        self.word2idx['<EOS>'] = 2
        self.word2idx['<UNK>'] = 3
        self.idx2word[0] = '<PAD>'
        self.idx2word[1] = '<BOS>'
        self.idx2word[2] = '<EOS>'
        self.idx2word[3] = '<UNK>'
        self.min_freq = min_freq
        self.vocab_size = 4

    def process_report(self, report):
        """对单个报告进行分词并返回词频计数，按单个字符分割"""
        return Counter(list(str(report)))

    def build_vocabulary(self, sentences, cache_path='vocab_cache.pkl'):
        """并行构建词汇表并支持缓存"""
        if os.path.exists(cache_path):
            print(f"从缓存加载词汇表: {cache_path}")
            self.load(cache_path)
            return

        print("并行构建词汇表...")
        unique_sentences = list(set(sentences))
        print(f"去重后报告数量: {len(unique_sentences)}")

        chunk_size = len(unique_sentences) // cpu_count() + 1
        sentence_chunks = [unique_sentences[i:i + chunk_size] for i in range(0, len(unique_sentences), chunk_size)]

        with Pool(processes=cpu_count()) as pool:
            chunk_counts = pool.map(self.process_report, unique_sentences)

        word_counts = sum(chunk_counts, Counter())

        for word, count in word_counts.items():
            if count >= self.min_freq:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1

        print(f"词汇表大小: {self.vocab_size}")
        self.save(cache_path)

    def text_to_sequence(self, text, max_length):
        """将文本转换为序列，按单个字符分割"""
        words = list(str(text))
        sequence = [self.word2idx['<BOS>']]
        for word in words:
            if len(sequence) < max_length - 1:
                sequence.append(self.word2idx.get(word, self.word2idx['<UNK>']))
        sequence.append(self.word2idx['<EOS>'])
        if len(sequence) < max_length:
            sequence.extend([self.word2idx['<PAD>']] * (max_length - len(sequence)))
        else:
            sequence = sequence[:max_length]
        return sequence

    def sequence_to_text(self, sequence):
        words = [self.idx2word.get(idx, '<UNK>') for idx in sequence]
        return ''.join(words)

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'word2idx': self.word2idx, 'idx2word': self.idx2word, 'vocab_size': self.vocab_size}, f,
                      ensure_ascii=False)

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.word2idx = data['word2idx']
        self.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        self.vocab_size = data['vocab_size']


@jit(nopython=True)
def largest_rectangle_in_histogram(heights, width):
    stack = []
    left_bounds = np.zeros(width, dtype=np.int32)
    right_bounds = np.full(width, width, dtype=np.int32)

    # 左边界
    for i in range(width):
        while stack and heights[stack[-1]] >= heights[i]:
            stack.pop()
        left_bounds[i] = stack[-1] + 1 if stack else 0
        stack.append(i)

    stack.clear()
    # 右边界
    for i in range(width - 1, -1, -1):
        while stack and heights[stack[-1]] >= heights[i]:
            stack.pop()
        right_bounds[i] = stack[-1] - 1 if stack else width - 1
        stack.append(i)

    max_area = 0
    best_left, best_right, best_height = 0, width - 1, 0
    for i in range(width):
        area = heights[i] * (right_bounds[i] - left_bounds[i] + 1)
        if area > max_area:
            max_area = area
            best_left = left_bounds[i]
            best_right = right_bounds[i]
            best_height = heights[i]

    return max_area, best_left, best_right, best_height


def get_largest_contour_rect(image: np.ndarray, bg_threshold: float = 0.05):
    """
    返回图像中最大轮廓的「最大内接矩形」坐标 (top, bottom, left, right)
    如果图像全黑或找不到轮廓，直接返回整张图的边界
    """
    if image.shape[-1] != 3:
        raise ValueError(f"Expected 3-channel RGB image, got shape {image.shape}")

    H, W = image.shape[:2]

    # 使用亮度阈值判断背景（任一通道大于等于阈值即为前景）
    brightness = np.max(image, axis=-1)
    mask = (brightness >= bg_threshold).astype(np.uint8) * 255

    if not np.any(mask):
        return 0, H - 1, 0, W - 1

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, H - 1, 0, W - 1

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    filled_mask = np.zeros((h, w), dtype=np.uint8)
    shifted_contour = largest_contour - [x, y]
    cv2.fillPoly(filled_mask, [shifted_contour], 255)

    actual_h, actual_w = filled_mask.shape
    heights = np.zeros(actual_w, dtype=np.int32)
    max_area = 0
    best_rect = None

    for row in range(actual_h):
        row_mask = filled_mask[row, :]
        heights = np.where(row_mask == 255, heights + 1, 0)

        area, left, right, rect_height = largest_rectangle_in_histogram(heights, actual_w)
        if area > max_area:
            max_area = area
            top = y + row - rect_height + 1
            bottom = y + row
            left_global = x + left
            right_global = x + right
            best_rect = (top, bottom, left_global, right_global)

    if best_rect is None:
        best_rect = (y, y + h - 1, x, x + w - 1)

    top, bottom, left, right = best_rect
    top = max(0, top)
    bottom = min(H - 1, bottom)
    left = max(0, left)
    right = min(W - 1, right)

    return top, bottom, left, right


class CropToLargestContour:
    """裁剪图像到最大轮廓内的最大内接矩形；如果失败则返回原图"""

    def __init__(self, bg_threshold: float = 0.05):
        self.bg_threshold = bg_threshold

    def __call__(self, image: Image.Image) -> Image.Image:
        try:
            image_np = np.array(image).astype(np.float32) / 255.0
            top, bottom, left, right = get_largest_contour_rect(image_np, self.bg_threshold)
            cropped = image_np[top:bottom + 1, left:right + 1]
            return Image.fromarray((cropped * 255).astype(np.uint8))
        except Exception as e:
            print(f"CropToLargestContour 失败，返回原图: {e}")
            return image


class RandomCropToLargestContour:
    """以一定概率执行最大内接矩形裁剪"""

    def __init__(self, prob=0.5, bg_threshold: float = 0.05):
        self.prob = prob
        self.crop_transform = CropToLargestContour(bg_threshold)

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.prob:
            return self.crop_transform(image)
        return image


class ReplaceImageFromFolderA:
    """以一定概率用文件夹A中的同名图片替换"""

    def __init__(self, folder_a, prob=0.5):
        self.folder_a = folder_a
        self.prob = prob
        self.image_files_a = {}
        for root, _, files in os.walk(folder_a):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    base = os.path.splitext(f)[0].lower()
                    self.image_files_a[base] = os.path.join(root, f)

    def __call__(self, img: Image.Image, filename: str):
        if random.random() < self.prob:
            base = os.path.splitext(os.path.basename(filename))[0].lower()
            replace_path = self.image_files_a.get(base)
            if replace_path and os.path.exists(replace_path):
                return Image.open(replace_path).convert('RGB')
        return img


class SharedAugmentDecision:
    """
    为一个样本生成统一的增强决策：
    - 以 select_replace_prob 概率决定是否使用 folder_a 的增强图替换（三个模态同时替换）
    - 否则，以 crop_prob 概率决定是否对三个模态同时进行随机裁剪
    支持变长输入：缺失模态传入 None，只对存在的模态应用增强，但决策共享。
    """

    def __init__(self,
                 folder_a_mod1=None, folder_a_mod2=None, folder_a_mod3=None,
                 select_replace_prob=0.1,
                 replace_prob=1.0,  # 替换时一定成功（只要文件存在）
                 crop_prob=0.5):
        self.select_replace_prob = select_replace_prob
        self.crop_prob = crop_prob
        # 为每个模态创建独立的替换器
        self.replace_mod1 = ReplaceImageFromFolderA(folder_a_mod1, prob=replace_prob) if folder_a_mod1 else None
        self.replace_mod2 = ReplaceImageFromFolderA(folder_a_mod2, prob=replace_prob) if folder_a_mod2 else None
        self.replace_mod3 = ReplaceImageFromFolderA(folder_a_mod3, prob=replace_prob) if folder_a_mod3 else None
        # 裁剪器（所有模态共享同一个概率）
        self.crop = RandomCropToLargestContour(prob=crop_prob)

    def __call__(self,
                 image_mod1: Optional[Image.Image] = None,
                 image_mod2: Optional[Image.Image] = None,
                 image_mod3: Optional[Image.Image] = None,
                 filename: str = "") -> Tuple[Optional[Image.Image], Optional[Image.Image], Optional[Image.Image]]:
        """
        输入：三个模态的原始 PIL Image（或 None 表示缺失）和 filename
        输出：增强后的三个 PIL Image（或 None，如果输入为 None）
        决策完全一致：使用共享随机值，只对非 None 模态应用。
        """
        # 收集输入图像
        images = [image_mod1, image_mod2, image_mod3]
        # 找到存在的模态索引
        valid_indices = [i for i, img in enumerate(images) if img is not None]

        # 如果全缺失，直接返回 None, None, None
        if not valid_indices:
            return None, None, None

        # 共享决策：决定是否替换
        do_replace = random.random() < self.select_replace_prob if (
                    self.replace_mod1 or self.replace_mod2 or self.replace_mod3) else False

        if do_replace:
            # 替换：只对有替换器且存在的模态应用
            output = list(images)  # 复制输入
            for i in valid_indices:
                if i == 0 and self.replace_mod1:
                    output[0] = self.replace_mod1(images[0], filename)
                elif i == 1 and self.replace_mod2:
                    output[1] = self.replace_mod2(images[1], filename)
                elif i == 2 and self.replace_mod3:
                    output[2] = self.replace_mod3(images[2], filename)
            return tuple(output)

        # 不替换：决定是否裁剪（共享决策）
        do_crop = random.random() < self.crop_prob

        output = list(images)
        if do_crop:
            for i in valid_indices:
                output[i] = self.crop(images[i])

        return tuple(output)


def apply_clahe_pil(pil_img, clip_limit=2.0, tile_grid_size=(8, 8)):
    if pil_img is None:
        return None
    # PIL → numpy (RGB)
    img_np = np.array(pil_img)
    # 转成 LAB 颜色空间，只对 L 通道做 CLAHE（保持颜色一致性）
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    # 转回 RGB
    rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb_clahe)


class PathologyDataset(Dataset):
    def __init__(self, image_dir_mod1, image_dir_mod2, image_dir_mod3, csv_path, vocab, max_seq_length=20,
                 transform=None, chunk_size=10000, display_num=5,
                 folder_a_mod1=None, folder_a_mod2=None, folder_a_mod3=None, replace_prob=0.5,
                 # 新增：模态缺失相关参数
                 missing_modalities=None,  # 可选值：None/['mod1']/['mod1','mod2'] 等，指定必缺失的模态
                 missing_prob=0.5,  # 随机缺失概率（对非必缺失模态生效）
                 allow_full_missing=False,
                 clahe_prob=0.3, clahe_clip_limit=3.0, clahe_tile_grid=(8, 8)):  # 是否允许三个模态全部缺失（不推荐，默认禁止）
        self.clahe_prob = clahe_prob
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid  # 统一命名
        self.image_dir_mod1 = image_dir_mod1
        self.image_dir_mod2 = image_dir_mod2
        self.image_dir_mod3 = image_dir_mod3
        self.csv_path = csv_path
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        self.chunk_size = chunk_size
        self.display_num = display_num
        self.folder_a_mod1 = folder_a_mod1
        self.folder_a_mod2 = folder_a_mod2
        self.folder_a_mod3 = folder_a_mod3
        # 新增：初始化模态缺失相关配置
        self.missing_modalities = missing_modalities if (
                    missing_modalities and isinstance(missing_modalities, list)) else []
        self.missing_prob = max(0.0, min(1.0, missing_prob))  # 限制概率在[0,1]区间
        self.allow_full_missing = allow_full_missing
        # 模态映射：方便后续批量处理
        self.modal_mapping = {
            'mod1': 0,
            'mod2': 1,
            'mod3': 2
        }
        self.all_modals = ['mod1', 'mod2', 'mod3']
        # 共同的变换（Resize 等）
        self.common_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        # 互斥增强，集中在 SharedAugmentDecision 中
        self.shared_augment = SharedAugmentDecision(
            folder_a_mod1=folder_a_mod1,
            folder_a_mod2=folder_a_mod2,
            folder_a_mod3=folder_a_mod3,
            select_replace_prob=0.4,  # 你可以调整这个概率
            replace_prob=1.0,  # 替换时强制执行
            crop_prob=0.5  # 裁剪概率（三个模态同时）
        )
        if csv_path.endswith('.xlsx') or csv_path.endswith('.xls'):
            csv_converted_path = csv_path.replace('.xlsx', '.csv').replace('.xls', '.csv')
            convert_excel_to_csv(csv_path, csv_converted_path)
            csv_path = csv_converted_path
        print(f"加载 CSV 文件: {csv_path}")
        self.df = pd.read_csv(csv_path, usecols=['filename', '编号'], encoding='utf-8')
        print(f"CSV 总行数: {len(self.df)}")
        if not {'filename', '编号'}.issubset(self.df.columns):
            raise ValueError("CSV 文件缺少 'filename' 或 '编号' 列")
        text_column = '编号'
        print(f"文本列名: {text_column}")
        self.df = self.df.dropna(subset=[text_column])
        print(f"移除空值后的行数: {len(self.df)}")
        self.df = self.df[self.df[text_column].str.strip() != '']
        print(f"移除空字符串后的行数: {len(self.df)}")
        print("\n递归检查图像文件...")
        all_image_files_mod1 = self._load_image_files(self.image_dir_mod1)
        all_image_files_mod2 = self._load_image_files(self.image_dir_mod2)
        all_image_files_mod3 = self._load_image_files(self.image_dir_mod3)
        self.all_image_paths_mod1 = []
        self.all_image_paths_mod2 = []
        self.all_image_paths_mod3 = []
        self.all_reports = []
        self.all_filenames = []
        print("\n匹配文件名和图像:")
        for _, row in self.df.iterrows():
            filename = row['filename']
            base_filename = str(filename).replace('.json', '').lower().strip()
            img_path_mod1 = all_image_files_mod1.get(base_filename)
            img_path_mod2 = all_image_files_mod2.get(base_filename)
            img_path_mod3 = all_image_files_mod3.get(base_filename)
            if img_path_mod1 and os.path.exists(img_path_mod1) and img_path_mod2 and os.path.exists(
                    img_path_mod2) and img_path_mod3 and os.path.exists(img_path_mod3):
                self.all_image_paths_mod1.append(img_path_mod1)
                self.all_image_paths_mod2.append(img_path_mod2)
                self.all_image_paths_mod3.append(img_path_mod3)
                self.all_filenames.append(base_filename)
                self.all_reports.append(row[text_column])
            else:
                print(f"未找到完整模态图像文件: {base_filename}")
        if len(self.all_image_paths_mod1) == 0:
            raise ValueError("未找到有效图像或文件名不匹配。")
        print(f"\n加载 {len(self.all_image_paths_mod1)} 张图像, {len(self.all_reports)} 份报告")
        for i in range(min(5, len(self.all_image_paths_mod1))):
            print(
                f"样本 {i}: 文件名={self.all_filenames[i]}, 报告={self.all_reports[i]}, 图像_mod1={self.all_image_paths_mod1[i]}, 图像_mod2={self.all_image_paths_mod2[i]}, 图像_mod3={self.all_image_paths_mod3[i]}")
        if len(set(self.all_reports)) < len(self.all_reports) * 0.5:
            print("警告: 报告内容多样性较低，建议检查数据集！")
        if self.vocab.vocab_size <= 4:
            print("构建词汇表...")
            self.vocab.build_vocabulary(self.all_reports, cache_path='vocab_cache.pkl')
            print(f"词汇表大小: {self.vocab.vocab_size}")
        self.total_chunks = (len(self.all_image_paths_mod1) + chunk_size - 1) // chunk_size
        self.update_chunk(0)
        self.display_images()
        # 新增：打印模态缺失配置信息
        print(
            f"\n模态缺失配置：必缺失模态={self.missing_modalities}, 随机缺失概率={self.missing_prob}, 允许全缺失={self.allow_full_missing}")

    def _load_image_files(self, image_dir):
        all_image_files = {}
        for root, _, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith('.png'):
                    base_filename = os.path.splitext(f)[0].lower()
                    all_image_files[base_filename] = os.path.join(root, f)
        return all_image_files

    def update_chunk(self, chunk_idx):
        """更新当前块的图像和报告列表"""
        self.current_chunk = chunk_idx
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, len(self.all_image_paths_mod1))
        self.image_paths_mod1 = self.all_image_paths_mod1[start_idx:end_idx]
        self.image_paths_mod2 = self.all_image_paths_mod2[start_idx:end_idx]
        self.image_paths_mod3 = self.all_image_paths_mod3[start_idx:end_idx]
        self.reports = self.all_reports[start_idx:end_idx]
        self.filenames = self.all_filenames[start_idx:end_idx]

    def display_images(self):
        print("\n显示样本图像:")
        # plt.figure(figsize=(15, 5))
        # for i in range(min(self.display_num, len(self.image_paths_mod1))):
        # img_path = self.image_paths_mod1[i]
        # filename = self.filenames[i]
        # image = Image.open(img_path).convert('RGB')
        # image_np = np.array(image)
        # plt.subplot(1, self.display_num, i + 1)
        # plt.imshow(image_np)
        # plt.title(f"文件名: {filename}")
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()

    def _get_missing_mask(self):
        """
        生成当前样本的模态缺失掩码
        返回：list，长度为3，对应[mod1, mod2, mod3]，True表示缺失，False表示存在
        """
        # 初始化缺失掩码：默认全部存在（False）
        missing_mask = [False] * 3
        # 第一步：处理必缺失模态
        for modal in self.missing_modalities:
            if modal in self.modal_mapping:
                missing_mask[self.modal_mapping[modal]] = True
        # 第二步：处理随机缺失（对非必缺失模态生效）
        for idx, modal in enumerate(self.all_modals):
            if not missing_mask[idx]:  # 仅对当前存在的模态进行随机缺失判断
                if torch.rand(1).item() < self.missing_prob:
                    missing_mask[idx] = True
        # 第三步：禁止全缺失（如果配置为不允许）
        if not self.allow_full_missing and all(missing_mask):
            # 随机选择一个模态恢复为存在
            restore_idx = torch.randint(0, 3, (1,)).item()
            missing_mask[restore_idx] = False
        return missing_mask

    def _get_zero_placeholder(self):
        """
        生成缺失模态的零张量占位符（与正常图像变换后的形状一致：3, 512, 512）
        返回：torch.Tensor，形状为(3, 512, 512)，值全为0
        """
        return torch.zeros((3, 512, 512), dtype=torch.float32)

    def __len__(self):
        return len(self.image_paths_mod1)

    def __getitem__(self, idx):
        # 1. 先获取所有模态的原始图像路径和文件名
        img_path_mod1 = self.image_paths_mod1[idx]
        img_path_mod2 = self.image_paths_mod2[idx]
        img_path_mod3 = self.image_paths_mod3[idx]
        filename = self.filenames[idx]
        # 2. 生成当前样本的模态缺失掩码
        missing_mask = self._get_missing_mask()
        # 3. 加载并处理每个模态（缺失则不加载 PIL，直接用 None，后续置零）
        image_mod1 = Image.open(img_path_mod1).convert('RGB') if not missing_mask[0] else None
        image_mod2 = Image.open(img_path_mod2).convert('RGB') if not missing_mask[1] else None
        image_mod3 = Image.open(img_path_mod3).convert('RGB') if not missing_mask[2] else None
        # 4. 对非缺失的模态应用互斥增强（缺失传入 None）
        aug_mod1, aug_mod2, aug_mod3 = self.shared_augment(image_mod1, image_mod2, image_mod3, filename)
        # 更新非缺失模态的增强结果
        if not missing_mask[0]:
            image_mod1 = aug_mod1
        if not missing_mask[1]:
            image_mod2 = aug_mod2
        if not missing_mask[2]:
            image_mod3 = aug_mod3

        # 对每个非缺失模态独立决定是否应用 CLAHE
        if not missing_mask[0] and torch.rand(1).item() < self.clahe_prob:
            image_mod1 = apply_clahe_pil(image_mod1, self.clahe_clip_limit, self.clahe_tile_grid)

        if not missing_mask[1] and torch.rand(1).item() < self.clahe_prob:
            image_mod2 = apply_clahe_pil(image_mod2, self.clahe_clip_limit, self.clahe_tile_grid)

        if not missing_mask[2] and torch.rand(1).item() < self.clahe_prob:
            image_mod3 = apply_clahe_pil(image_mod3, self.clahe_clip_limit, self.clahe_tile_grid)

        # 5. 对非缺失的模态应用共同变换（缺失用零张量）
        if not missing_mask[0]:
            image_mod1 = self.common_transform(image_mod1)
        else:
            image_mod1 = self._get_zero_placeholder()
        if not missing_mask[1]:
            image_mod2 = self.common_transform(image_mod2)
        else:
            image_mod2 = self._get_zero_placeholder()
        if not missing_mask[2]:
            image_mod3 = self.common_transform(image_mod3)
        else:
            image_mod3 = self._get_zero_placeholder()
        # 6. 处理报告序列（保持原有逻辑不变）
        report = self.reports[idx]
        target_sequence = torch.tensor(
            self.vocab.text_to_sequence(report, self.max_seq_length),
            dtype=torch.long
        )
        # 7. （可选）返回缺失掩码，方便后续分析（如需保持原输出格式，可移除该返回值）
        missing_mask_tensor = torch.tensor(missing_mask, dtype=torch.bool)
        # 输出：保持原格式 + 可选缺失掩码（如需兼容原有代码，可返回 image_mod1, image_mod2, image_mod3, target_sequence, filename）
        return image_mod1, image_mod2, image_mod3, target_sequence, filename

    def save_vocab(self, path):
        self.vocab.save(path)

    def load_vocab(self, path):
        self.vocab.load(path)


# 坐标注意力模块
class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mid_channels = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.bn = nn.InstanceNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h_pool = self.pool_h(x).permute(0, 1, 3, 2)
        w_pool = self.pool_w(x)
        y = torch.cat([h_pool, w_pool], dim=2)
        y = self.conv1(y)
        y = self.bn(y)
        y = self.act(y)

        actual_size = y.shape[2]
        expected_size = H + W
        if actual_size != expected_size:
            if actual_size < 2:
                return x
            h_size = int(actual_size * H / (H + W)) if H + W > 0 else actual_size // 2
            w_size = actual_size - h_size
        else:
            h_size, w_size = H, W

        h_attn, w_attn = torch.split(y, [h_size, w_size], dim=2)
        h_attn = self.conv_h(h_attn.permute(0, 1, 3, 2))
        w_attn = self.conv_w(w_attn)
        attn = torch.sigmoid(h_attn) * torch.sigmoid(w_attn)
        return x * attn


# 模型类
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(out_channels)
        self.coord_attn = CoordinateAttention(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.coord_attn(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CustomResNet(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_channels=64):
        super(CustomResNet, self).__init__()
        self.in_channels = num_channels
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.InstanceNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, num_channels, layers[0], stride=2)
        self.layer2 = self._make_layer(block, num_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, num_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, num_channels * 8, layers[3], stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_channels * 8 * block.expansion, 2048)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ImageToTextTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, max_seq_length=100,
                 image_feature_dim=2048, vocab=None):
        super(ImageToTextTransformer, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.image_feature_dim = image_feature_dim
        self.vocab = vocab  # 保存 vocab 用于查找数字 token

        # 三个模态的特征提取器
        self.resnet_mod1 = CustomResNet()
        self.resnet_mod2 = CustomResNet()
        self.resnet_mod3 = CustomResNet()

        # 新增：投影三个模态特征到 d_model 维度，以便注意力融合
        self.proj_mod1 = nn.Linear(image_feature_dim, d_model)
        self.proj_mod2 = nn.Linear(image_feature_dim, d_model)
        self.proj_mod3 = nn.Linear(image_feature_dim, d_model)

        # 新增：位置编码 for 模态序列 (假设固定3个模态)
        self.mod_pos_encoder = nn.Parameter(torch.zeros(1, 3, d_model))

        # 新增：交叉注意力融合模块，使用 Transformer Encoder 来融合三个模态特征
        # 这里用一个小型 Transformer Encoder (1-2层) 来允许模态间交互
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward // 4,  # 缩小以避免过拟合
            dropout=0.1,
            batch_first=True
        )
        self.fusion_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2  # 用2层来捕捉交互，足够但不复杂
        )

        # 最终投影到 decoder 维度 (现在作用于序列)
        self.image_projection = nn.Linear(d_model, d_model)

        # 文本侧 (不变)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

        # ==================== 新增：记录数字 token 的索引 ====================
        self.register_buffer('digit_indices', torch.tensor([], dtype=torch.long))
        if vocab is not None:
            digits = []
            for d in '0123456789':
                idx = vocab.word2idx.get(d, -1)
                if idx != -1:
                    digits.append(idx)
            if digits:
                self.digit_indices = torch.tensor(digits, dtype=torch.long)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # 对于新加的投影，使用标准初始化

    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask.to(device)

    def extract_image_features(self, images_mod1, images_mod2, images_mod3):
        device = next(self.parameters()).device
        images_mod1 = images_mod1.to(device)
        images_mod2 = images_mod2.to(device)
        images_mod3 = images_mod3.to(device)

        # 提取原始特征 [B, 2048]
        feat_mod1 = self.resnet_mod1(images_mod1)
        feat_mod2 = self.resnet_mod2(images_mod2)
        feat_mod3 = self.resnet_mod3(images_mod3)

        # 投影到 d_model [B, d_model]
        proj_mod1 = self.proj_mod1(feat_mod1)
        proj_mod2 = self.proj_mod2(feat_mod2)
        proj_mod3 = self.proj_mod3(feat_mod3)

        # 堆叠成序列 [B, 3, d_model]
        feat_seq = torch.stack([proj_mod1, proj_mod2, proj_mod3], dim=1)

        # 加位置编码
        feat_seq = feat_seq + self.mod_pos_encoder.to(device)

        # 通过 Transformer Encoder 融合 (允许每个模态 attend 到其他)
        fused_seq = self.fusion_encoder(feat_seq)  # [B, 3, d_model]

        # 不再取平均，直接投影序列
        fused_seq = self.image_projection(fused_seq)  # [B, 3, d_model]

        return fused_seq  # 返回多 token 序列

    def forward(self, images_mod1, images_mod2, images_mod3, tgt, tgt_mask=None):
        device = tgt.device
        # 提取并融合多模态特征
        fused_features = self.extract_image_features(images_mod1, images_mod2, images_mod3)
        memory = fused_features  # [B, 3, d_model]  # 多 token memory

        # 文本嵌入 + 位置编码
        tgt_emb = self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float, device=device))
        tgt_emb = tgt_emb + self.pos_encoder[:, :tgt.size(1), :].to(device)

        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), device)

        output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask
        )  # [B, S_tgt, D]

        output = self.fc_out(output)  # [B, S_tgt, vocab_size]

        # ==================== 新增：前缀位置强制禁止生成数字 ====================
        # 只在训练时生效，且序列足够长时（至少有第5个位置）
        # if self.training and self.digit_indices.numel() > 0 and output.size(1) >= 5:
        #     # 创建数字 token 的 mask
        #     digit_mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=output.device)
        #     digit_mask[self.digit_indices] = True
        #     # 用一个很大的负数压制数字 token 的 logits（避免 -inf 导致 nan）
        #     LARGE_NEG = -100.0
        #     # 对第 2~5 个 token（索引 1:5）位置的数字 logits 施加惩罚
        #     output[:, 1:5, digit_mask] = LARGE_NEG

        return output

    def generate(self, images_mod1, images_mod2, images_mod3, max_length=50, start_token=1, end_token=2,
                 top_k=50, temperature=1.0, use_greedy=False):
        self.eval()
        device = next(self.parameters()).device
        batch_size = images_mod1.size(0)
        fused_features = self.extract_image_features(images_mod1, images_mod2, images_mod3)
        memory = fused_features  # [B, 3, d_model]  # 多 token memory
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        all_logits = []
        with torch.no_grad():
            for _ in range(max_length):
                output = self.forward(images_mod1, images_mod2, images_mod3, generated)
                all_logits.append(output[:, -1:, :])
                next_token_logits = output[:, -1, :] / temperature
                if use_greedy:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    probs = F.softmax(next_token_logits, dim=-1)
                    top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
                    next_token = torch.multinomial(top_k_probs, num_samples=1)
                    next_token = top_k_indices.gather(-1, next_token)
                generated = torch.cat([generated, next_token], dim=1)
                if torch.all(next_token.squeeze(1) == end_token):
                    break
        generated_logits = torch.cat(all_logits, dim=1)
        return generated, generated_logits


def convert_excel_to_csv(excel_path, csv_path):
    if not os.path.exists(csv_path):
        print(f"将 Excel 文件 {excel_path} 转换为 CSV...")
        df = pd.read_excel(excel_path)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"转换完成，CSV 文件保存至 {csv_path}")


def get_token_length(text):
    """对文本进行分词并返回分词后的长度，按单个字符分割"""
    return len(list(str(text)))


def calculate_max_seq_length(csv_path, extra_padding=2, cache_path='max_seq_length.pkl'):
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    if csv_path.endswith('.xlsx') or csv_path.endswith('.xls'):
        csv_converted_path = csv_path.replace('.xlsx', '.csv').replace('.xls', '.csv')
        convert_excel_to_csv(csv_path, csv_converted_path)
        csv_path = csv_converted_path

    try:
        max_length = 0
        for chunk in pd.read_csv(csv_path, usecols=['编号'], chunksize=10000, encoding='utf-8'):
            reports = chunk['编号'].dropna().astype(str).drop_duplicates()
            with Pool(processes=cpu_count()) as pool:
                lengths = pool.map(get_token_length, reports)
            max_length = max(max_length, max(lengths, default=0))

        result = max_length + extra_padding
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)
        return result
    except Exception as e:
        raise ValueError(f"计算 max_seq_length 失败: {str(e)}")


def sequence_level_loss(generated, targets, vocab, tokenizer, bert_model, device, alpha=0.7, beta=0.3):
    generated_tokens, generated_logits = generated
    generated_tokens = generated_tokens.to(device)
    generated_logits = generated_logits.to(device)
    targets = targets.to(device)

    max_len = min(generated_logits.size(1), targets.size(1))
    generated_logits = generated_logits[:, :max_len, :]
    targets = targets[:, :max_len]

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<PAD>'], reduction='mean')
    token_loss = criterion(generated_logits.reshape(-1, vocab.vocab_size), targets.reshape(-1))

    gen_texts = [vocab.sequence_to_text(g.cpu().numpy()) for g in generated_tokens]
    tgt_texts = [vocab.sequence_to_text(t.cpu().numpy()) for t in targets]

    gen_inputs = tokenizer(gen_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    tgt_inputs = tokenizer(tgt_texts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        gen_embeds = bert_model(**gen_inputs).last_hidden_state.mean(dim=1)
        tgt_embeds = bert_model(**tgt_inputs).last_hidden_state.mean(dim=1)

    semantic_loss = 1 - F.cosine_similarity(gen_embeds, tgt_embeds).mean()

    total_loss = alpha * token_loss + beta * semantic_loss
    return total_loss, token_loss, semantic_loss


# ============================ 前缀加权相关 ============================
PREFIX_PATTERN = re.compile(r'^[A-Z]{4}\d{5}$')  # 严格匹配 4大写字母 + 5数字


def get_prefix_weight_mask(targets, vocab, boost=8.0):
    """
    targets: (B, L) LongTensor，包含 <BOS> ... <EOS>
    返回: (B, L-1) FloatTensor，权重 mask
    """
    batch_size, seq_len = targets.shape
    weights = torch.ones(batch_size, seq_len - 1, dtype=torch.float, device=targets.device)

    special_tokens = {
        vocab.word2idx.get('<PAD>', 0),
        vocab.word2idx.get('<BOS>', 1),
        vocab.word2idx.get('<EOS>', 2),
        vocab.word2idx.get('<UNK>', 3)
    }

    for b in range(batch_size):
        seq = targets[b].cpu().numpy()
        tokens = [
            vocab.idx2word.get(idx, '') for idx in seq
            if idx not in special_tokens
        ]
        text = ''.join(tokens).strip()

        if PREFIX_PATTERN.match(text):  # 整条报告以 XXXX12345 开头
            # 前4个字母对应输出位置的 index 1,2,3,4（因为 input 是[:-1]，output 预测 [1:]）
            weights[b, 1:5] = boost

    return weights


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ============================ 路径配置 ============================
    image_dir_mod1 = r"/root/autodl-tmp/CT-T2/CT-T2"
    image_dir_mod2 = r"/root/autodl-tmp/ADC/ADC"
    image_dir_mod3 = r"/root/autodl-tmp/DWI/DWI"
    csv_path = r"/root/autodl-tmp/merged_output.csv"
    folder_a_mod1 = r"/root/autodl-tmp/CT-T2增强/CT-T2增强"
    folder_a_mod2 = r"/root/autodl-tmp/ADC_增强/ADC_增强"
    folder_a_mod3 = r"/root/autodl-tmp/DWI_增强/DWI_增强"
    vocab_save_path = r"/root/autodl-tmp/vocab.json"
    model_save_path = r'/root/autodl-tmp/model_多模态2.pth'

    # ============================ 超参数 ============================
    batch_size = 32
    accumulation_steps = 16  # 有效 batch_size = 512
    num_epochs = 4000
    learning_rate = 1e-4
    warmup_epochs = 4000
    alpha = 1.0  # CE 损失权重
    beta = 0.2  # sequence-level 损失权重
    temperature = 0.7
    PREFIX_BOOST = 5.0  # 前缀位置权重提升，建议 5.0~10.0

    # ============================ 词汇表 & 数据集 ============================
    max_seq_length = calculate_max_seq_length(csv_path, extra_padding=2)
    print(f"计算得到的 max_seq_length: {max_seq_length}")

    vocab = Vocabulary(min_freq=1)
    dataset = PathologyDataset(
        image_dir_mod1=image_dir_mod1,
        image_dir_mod2=image_dir_mod2,
        image_dir_mod3=image_dir_mod3,
        csv_path=csv_path,
        vocab=vocab,
        max_seq_length=max_seq_length,
        chunk_size=1000000000,
        folder_a_mod1=folder_a_mod1,
        folder_a_mod2=folder_a_mod2,
        folder_a_mod3=folder_a_mod3
    )
    print(f"数据集大小: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=14,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=True,
        multiprocessing_context='fork'
    )

    dataset.save_vocab(vocab_save_path)
    print(f"词汇表已保存至: {vocab_save_path}，大小: {vocab.vocab_size}")

    # ============================ 模型 ============================
    model = ImageToTextTransformer(
        vocab_size=vocab.vocab_size,
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        max_seq_length=max_seq_length,
        image_feature_dim=2048,
        vocab=vocab  # 必须传入，用于禁止数字 + 前缀加权
    ).to(device)

    # 加载预训练权重（可选）
    model.load_state_dict(torch.load(r'model_多模态2.pth', map_location=device), strict=False)
    print("已加载预训练权重 model_L3.pth")

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 用于加权 CE 的 criterion（reduction='none'）
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<PAD>'], reduction='none')

    # # BERT 用于语义损失
    # cache_dir = r"/media/lenovo/6ED3FFE79A41910F/CT-HE测试/人济mask/transformer/bert-base-chinese"
    # tokenizer = AutoTokenizer.from_pretrained(cache_dir, local_files_only=True)
    # bert_model = AutoModel.from_pretrained(cache_dir, local_files_only=True).to(device)
    # bert_model.eval()

    # 混合精度
    scaler = amp.GradScaler() if device.type == 'cuda' else None

    total_batches = len(dataloader)
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\n=== 训练轮次 {epoch}/{num_epochs} ===")
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")

        for batch_idx, (images_mod1, images_mod2, images_mod3, targets, patient_ids) in enumerate(pbar):
            global_step += 1

            images_mod1 = images_mod1.to(device, non_blocking=True)
            images_mod2 = images_mod2.to(device, non_blocking=True)
            images_mod3 = images_mod3.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with amp.autocast(enabled=(device.type == 'cuda')):
                # Teacher-forcing 前向
                output = model(images_mod1, images_mod2, images_mod3, targets[:, :-1])  # (B, L-1, V)

                # ==================== 加权 CE 损失 ====================
                pos_weights = get_prefix_weight_mask(targets, vocab, boost=PREFIX_BOOST)  # (B, L-1)

                loss_per_token = criterion(
                    output.reshape(-1, vocab.vocab_size),
                    targets[:, 1:].reshape(-1)
                )  # (B*(L-1),)

                loss_per_token = loss_per_token.view(batch_size, -1)  # (B, L-1)
                weighted_loss = loss_per_token * pos_weights
                ce_loss = weighted_loss.mean()

                # ==================== Sequence-level loss（可选）===================
                seq_loss = token_loss = semantic_loss = torch.tensor(0.0, device=device)
                # if epoch > warmup_epochs:
                #     with torch.no_grad():
                #         generated, _ = model.generate(
                #             images_mod1, images_mod2, images_mod3,
                #             max_length=max_seq_length,
                #             start_token=vocab.word2idx['<BOS>'],
                #             end_token=vocab.word2idx['<EOS>'],
                #             top_k=33,
                #             temperature=temperature,
                #             use_greedy=False
                #         )
                #     seq_loss, token_loss, semantic_loss = sequence_level_loss(
                #         generated, targets, vocab, tokenizer, bert_model, device,
                #         alpha=0.7, beta=0.3
                #     )

                loss = alpha * ce_loss + beta * seq_loss
                loss = loss / accumulation_steps

            # 反向传播
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * accumulation_steps

            # 梯度累积更新
            if global_step % accumulation_steps == 0 or (batch_idx + 1) == total_batches:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()

            pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'ce': f'{ce_loss.item():.4f}',
                'seq': f'{seq_loss.item():.4f}'
            })

        avg_loss = epoch_loss / total_batches
        print(f"Epoch {epoch} 完成，平均损失: {avg_loss:.4f}")

        # 保存模型（建议后期改为按验证指标保存最佳）
        torch.save(model.state_dict(), model_save_path)

    print("训练完成！")

    # ============================ 推理示例 ============================
    print("\n=== 推理示例（前几个样本）===")
    model.eval()
    with torch.no_grad():
        for images_mod1, images_mod2, images_mod3, targets, patient_ids in dataloader:
            images_mod1 = images_mod1.to(device)
            images_mod2 = images_mod2.to(device)
            images_mod3 = images_mod3.to(device)

            with amp.autocast(enabled=(device.type == 'cuda')):
                generated, _ = model.generate(
                    images_mod1, images_mod2, images_mod3,
                    max_length=max_seq_length,
                    start_token=vocab.word2idx['<BOS>'],
                    end_token=vocab.word2idx['<EOS>'],
                    top_k=33,
                    temperature=temperature
                )

            for i in range(generated.shape[0]):
                gen_text = vocab.sequence_to_text(generated[i].cpu().numpy())
                gt_text = vocab.sequence_to_text(targets[i].cpu().numpy())
                print(f"患者ID: {patient_ids[i]:<10} 生成: {gen_text}")
                print(f"{'':<18} 真实: {gt_text}\n")
            break


if __name__ == "__main__":
    main()