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

# 设置Hugging Face国内镜像源（魔搭社区镜像）
os.environ["HF_ENDPOINT"] = "https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models"


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

    for i in range(width):
        while stack and heights[stack[-1]] >= heights[i]:
            stack.pop()
        left_bounds[i] = stack[-1] + 1 if stack else 0
        stack.append(i)

    stack.clear()
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


# --------------------- 2. 核心裁剪函数（已升级） ---------------------
def get_largest_contour_rect(image: np.ndarray, bg_threshold: float = 0.05):
    """
    image: [H, W, 3] float32, 范围 0~1
    bg_threshold: 小于这个值视为背景（黑色）
    """
    if image.shape[-1] != 3:
        raise ValueError(f"Expected 3-channel RGB image, got shape {image.shape}")

    H, W = image.shape[:2]

    # 关键修改：用亮度阈值判断背景（更鲁棒！）
    brightness = np.max(image, axis=-1)  # 取 RGB 最大通道作为亮度
    mask = (brightness >= bg_threshold).astype(np.uint8) * 255  # >= 0.05 才算前景

    if not np.any(mask):
        return 0, H - 1, 0, W - 1

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, H - 1, 0, W - 1

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # 填充实心（防空洞）
    filled_mask = np.zeros((h, w), dtype=np.uint8)
    shifted_contour = largest_contour - [x, y]
    cv2.fillPoly(filled_mask, [shifted_contour], 255)

    actual_h, actual_w = filled_mask.shape
    heights = np.zeros(actual_w, dtype=np.int32)
    max_area = 0
    best_rect = (y, y + h - 1, x, x + w - 1)  # 默认回退到 bounding box

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

    top, bottom, left, right = best_rect
    top = max(0, top)
    bottom = min(H - 1, bottom)
    left = max(0, left)
    right = min(W - 1, right)

    return top, bottom, left, right


# --------------------- 3. 增强类（带可视化） ---------------------
class CropToLargestContour:
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

    def __init__(self, prob=0.5):
        self.prob = prob
        self.crop_transform = CropToLargestContour()

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


# 新增的互斥增强类
class MutualExclusiveAugment:
    def __init__(self, folder_a, select_replace_prob=0.5, replace_prob=0.5, crop_prob=0.5):
        self.select_replace_prob = select_replace_prob
        self.replace = ReplaceImageFromFolderA(folder_a, prob=replace_prob) if folder_a else None
        self.crop = RandomCropToLargestContour(prob=crop_prob)

    def __call__(self, image, filename):
        if self.replace and random.random() < self.select_replace_prob:
            # print("调试：选择使用 ReplaceImageFromFolderA 增强")
            return self.replace(image, filename)
        else:
            # print("调试：选择使用 RandomCropToLargestContour 增强")
            return self.crop(image)


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
    def __init__(self, image_dir, csv_path, vocab, max_seq_length=20, transform=None, chunk_size=10000, display_num=5,
                 folder_a=None, replace_prob=0.5, clahe_prob=0.3, clahe_clip_limit=2.0, clahe_tile_size=(8, 8)):
        self.clahe_prob = clahe_prob  # 每个模态独立应用 CLAHE 的概率
        self.clahe_clip = clahe_clip_limit
        self.clahe_tile = clahe_tile_size
        self.image_dir = image_dir
        self.csv_path = csv_path
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        self.chunk_size = chunk_size
        self.display_num = display_num
        self.folder_a = folder_a

        # 共同的变换（Resize 等）
        self.common_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS),
            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomVerticalFlip(0.5),
            # transforms.RandomRotation(30),
            # transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 互斥增强，集中在 MutualExclusiveAugment 中
        self.mutual_augment = MutualExclusiveAugment(
            folder_a,
            select_replace_prob=0.5,  # 选择 Replace 的概率
            replace_prob=0.5,  # Replace 内部替换概率
            crop_prob=0.5  # Crop 内部裁剪概率
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
        all_image_files = {}
        for root, _, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith('.png'):
                    base_filename = os.path.splitext(f)[0].lower()
                    all_image_files[base_filename] = os.path.join(root, f)

        self.all_image_paths = []
        self.all_reports = []
        self.all_filenames = []
        print("\n匹配文件名和图像:")
        for _, row in self.df.iterrows():
            filename = row['filename']
            base_filename = str(filename).replace('.json', '').lower().strip()
            img_path = all_image_files.get(base_filename)
            if img_path and os.path.exists(img_path):
                self.all_image_paths.append(img_path)
                self.all_filenames.append(base_filename)
                self.all_reports.append(row[text_column])
            else:
                print(f"未找到图像文件: {base_filename}")

        if len(self.all_image_paths) == 0:
            raise ValueError("未找到有效图像或文件名不匹配。")

        print(f"\n加载 {len(self.all_image_paths)} 张图像, {len(self.all_reports)} 份报告")
        for i in range(min(5, len(self.all_image_paths))):
            print(
                f"样本 {i}: 文件名={self.all_filenames[i]}, 报告={self.all_reports[i]}, 图像={self.all_image_paths[i]}")

        if len(set(self.all_reports)) < len(self.all_reports) * 0.5:
            print("警告: 报告内容多样性较低，建议检查数据集！")

        if self.vocab.vocab_size <= 4:
            print("构建词汇表...")
            self.vocab.build_vocabulary(self.all_reports, cache_path='vocab_cache.pkl')
            print(f"词汇表大小: {self.vocab.vocab_size}")

        self.total_chunks = (len(self.all_image_paths) + chunk_size - 1) // chunk_size
        self.update_chunk(0)

        self.display_images()

    def update_chunk(self, chunk_idx):
        """更新当前块的图像和报告列表"""
        self.current_chunk = chunk_idx
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, len(self.all_image_paths))
        self.image_paths = self.all_image_paths[start_idx:end_idx]
        self.reports = self.all_reports[start_idx:end_idx]
        self.filenames = self.all_filenames[start_idx:end_idx]

    def display_images(self):
        print("\n显示样本图像:")
        # plt.figure(figsize=(15, 5))
        # for i in range(min(self.display_num, len(self.image_paths))):
        #     img_path = self.image_paths[i]
        #     filename = self.filenames[i]
        #     image = Image.open(img_path).convert('RGB')
        #     image_np = np.array(image)
        #     plt.subplot(1, self.display_num, i + 1)
        #     plt.imshow(image_np)
        #     plt.title(f"文件名: {filename}")
        #     plt.axis('off')
        # plt.tight_layout()
        # plt.show()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        filename = self.filenames[idx]

        # 先应用互斥增强（每次样本独立随机选择）
        image = self.mutual_augment(image, filename)

        if torch.rand(1).item() < self.clahe_prob:
            image = apply_clahe_pil(
                image,
                clip_limit=self.clahe_clip,
                tile_grid_size=self.clahe_tile
            )

        # 再应用共同变换
        image = self.common_transform(image)

        report = self.reports[idx]
        target_sequence = torch.tensor(
            self.vocab.text_to_sequence(report, self.max_seq_length),
            dtype=torch.long
        )
        return image, target_sequence, filename

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
                 image_feature_dim=2048, vocab=None):  # 新增 vocab 参数
        super(ImageToTextTransformer, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.image_feature_dim = image_feature_dim
        self.vocab = vocab  # 保存 vocab 用于查找数字 token

        self.resnet = CustomResNet()
        self.image_projection = nn.Linear(image_feature_dim, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_length, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1
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

    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask.to(device)

    def extract_image_features(self, images):
        device = next(self.parameters()).device
        if isinstance(images, list):
            images = torch.stack(images).to(device)
        else:
            images = images.to(device)
        with torch.no_grad():
            features = self.resnet(images)
        return features

    def forward(self, images, tgt, tgt_mask=None):
        device = tgt.device
        image_features = self.resnet(images)
        image_features = self.image_projection(image_features).unsqueeze(1)  # (B, 1, D)

        tgt_embed = self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float, device=device))
        tgt_embed = tgt_embed + self.pos_encoder[:, :tgt.size(1), :].to(device)

        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), device)

        output = self.transformer_decoder(
            tgt_embed.transpose(0, 1),
            image_features.transpose(0, 1),
            tgt_mask=tgt_mask
        ).transpose(0, 1)  # (B, L, D)

        output = self.fc_out(output)  # (B, L, V)

        # ==================== 新增：前缀位置强制禁止生成数字 ====================
        # 只在训练时生效，且序列足够长时
        #         if self.training and self.digit_indices.numel() > 0 and output.size(1) >= 5:
        #             digit_mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=output.device)
        #             digit_mask[self.digit_indices] = True

        #             # 用一个很大的负数代替 -inf，避免 nan
        #             LARGE_NEG = -100  # 足够大，softmax 后概率接近 0
        #             output[:, 1:5, digit_mask] = LARGE_NEG

        return output

    def generate(self, images, max_length=50, start_token=1, end_token=2, top_k=50, temperature=1.0, use_greedy=False):
        self.eval()
        device = next(self.parameters()).device
        batch_size = len(images) if isinstance(images, list) else images.size(0)
        image_features = self.extract_image_features(images)
        image_features = self.image_projection(image_features).unsqueeze(1)
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        all_logits = []
        with torch.no_grad():
            for _ in range(max_length):
                output = self.forward(images, generated)
                all_logits.append(output[:, -1:, :])
                next_token_logits = output[:, -1, :] / temperature
                if use_greedy:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    top_k_probs, top_k_indices = torch.topk(F.softmax(next_token_logits, dim=-1), top_k, dim=-1)
                    next_token = torch.multinomial(top_k_probs, num_samples=1)
                    next_token = top_k_indices.gather(-1, next_token)
                generated = torch.cat([generated, next_token], dim=1)
                if torch.all(next_token == end_token):
                    break
                del output
                torch.cuda.empty_cache()
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


PREFIX_PATTERN = re.compile(r'^[A-Z]{4}\d{5}$')  # 更简洁，也可以捕获但不需要 group


def get_prefix_weight_mask(targets, vocab, boost=8.0):
    batch_size, seq_len = targets.shape
    weights = torch.ones(batch_size, seq_len - 1, dtype=torch.float, device=targets.device)

    for b in range(batch_size):
        seq = targets[b].cpu().numpy()
        tokens = [vocab.idx2word.get(idx, '') for idx in seq
                  if idx not in {vocab.word2idx.get('<PAD>', 0),
                                 vocab.word2idx.get('<BOS>', 1),
                                 vocab.word2idx.get('<EOS>', 2),
                                 vocab.word2idx.get('<UNK>', 3)}]
        text = ''.join(tokens).strip()

        if PREFIX_PATTERN.match(text):  # 严格匹配 4字母+5数字
            # 加权前 4 个字母的预测位置 → output 的 index 1,2,3,4
            weights[b, 1:5] = boost

    return weights


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ====================== 混合精度相关 ======================
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("已启用混合精度训练 (AMP)")

    # ====================== 路径与超参数 ======================
    image_dir = r"/root/autodl-tmp/CT-T2/CT-T2"
    csv_path = r"/root/autodl-tmp/merged_output.csv"
    vocab_save_path = r"/root/autodl-tmp/vocab.json"
    batch_size = 128
    accumulation_steps = 4
    num_epochs = 4000
    learning_rate = 0.0001
    warmup_epochs = 4000
    alpha = 1.0  # CE 损失权重
    beta = 0.2  # seq_loss 权重（如果需要）
    temperature = 0.7

    # 前缀加权超参数（关键！）
    PREFIX_BOOST = 8.0  # 建议从 5.0 ~ 10.0 尝试，越大前缀学得越快

    # 计算最大序列长度
    max_seq_length = calculate_max_seq_length(csv_path, extra_padding=2)
    print(f"计算得到的 max_seq_length: {max_seq_length}")

    # 构建词汇表与数据集
    vocab = Vocabulary(min_freq=1)
    dataset = PathologyDataset(
        image_dir=image_dir,
        csv_path=csv_path,
        vocab=vocab,
        max_seq_length=max_seq_length,
        chunk_size=1000000000,
        folder_a=r'/root/autodl-tmp/CT-T2增强/CT-T2增强'
    )
    print(f"数据集大小: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=True
    )

    dataset.save_vocab(vocab_save_path)
    print(f"词汇表已保存至: {vocab_save_path}")
    print(f"词汇表大小: {vocab.vocab_size}")

    # 模型加载
    model = ImageToTextTransformer(
        vocab_size=vocab.vocab_size,
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        max_seq_length=max_seq_length,
        image_feature_dim=2048,
        vocab=vocab
    ).to(device)

    model.load_state_dict(torch.load(r'/root/autodl-tmp/model12.pth'), strict=False)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # BERT（语义损失可选）
    cache_dir = r"/root/autodl-tmp/transformer/bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-chinese",
        cache_dir=cache_dir,
        local_files_only=True
    )
    bert_model = AutoModel.from_pretrained(
        "bert-base-chinese",
        cache_dir=cache_dir,
        local_files_only=True
    ).to(device)
    bert_model.eval()

    total_batches_per_epoch = len(dataloader)

    for epoch in range(num_epochs):
        print(f"\n=== 训练轮次 {epoch + 1}/{num_epochs} ===")
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

        for step, (images, targets, patient_ids) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(images, targets[:, :-1])  # (B, L-1, V)

                # ==================== 位置加权 CE 损失 ====================
                # 获取前缀位置权重 mask
                pos_weights = get_prefix_weight_mask(targets, vocab, boost=PREFIX_BOOST)

                # 使用 reduction='none' 计算每个 token 的损失
                criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<PAD>'], reduction='none')
                loss_per_token = criterion(
                    output.reshape(-1, vocab.vocab_size),
                    targets[:, 1:].reshape(-1)
                )  # (B*(L-1),)

                # reshape 回 (B, L-1)
                loss_per_token = loss_per_token.view(output.size(0), -1)

                # 应用位置权重
                weighted_loss = loss_per_token * pos_weights

                # 取平均
                ce_loss = weighted_loss.mean()

                # ==================== Sequence-level loss（可选）===================
                seq_loss = token_loss = semantic_loss = torch.tensor(0.0, device=device)
                if epoch >= warmup_epochs:
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            generated = model.generate(
                                images,
                                max_length=max_seq_length,
                                start_token=vocab.word2idx['<BOS>'],
                                end_token=vocab.word2idx['<EOS>'],
                                top_k=33,
                                temperature=temperature
                            )
                    seq_loss, token_loss, semantic_loss = sequence_level_loss(
                        generated, targets, vocab, tokenizer, bert_model, device,
                        alpha=0.7, beta=0.3
                    )

                loss = alpha * ce_loss + beta * seq_loss
                loss = loss / accumulation_steps

            # 反向传播
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            real_batch_loss = loss.item() * accumulation_steps
            total_loss += real_batch_loss

            # 梯度累积更新
            if (step + 1) % accumulation_steps == 0 or (step + 1) == total_batches_per_epoch:
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
                'Loss': f'{real_batch_loss:.4f}',
                'CE': f'{ce_loss.item():.4f}',
                'Token': f'{token_loss.item():.4f}',
                'Sem': f'{semantic_loss.item():.4f}',
                'GPU': f'{torch.cuda.memory_allocated(device) / 1e9:.2f}GB' if use_amp else 'N/A'
            })

        avg_loss = total_loss / total_batches_per_epoch
        print(f"轮次 {epoch + 1}/{num_epochs} 完成，平均损失: {avg_loss:.4f}")

        # 每轮保存（建议改成带 epoch 的文件名防止覆盖）
        torch.save(model.state_dict(), r'/root/autodl-tmp/model13.pth')

    print("训练完成！")

    # ====================== 简单推理验证 ======================
    model.eval()
    with torch.no_grad():
        for images, targets, patient_ids in dataloader:
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                generated, _ = model.generate(
                    images,
                    max_length=max_seq_length,
                    start_token=vocab.word2idx['<BOS>'],
                    end_token=vocab.word2idx['<EOS>'],
                    top_k=33,
                    temperature=temperature
                )
            for i in range(min(5, generated.shape[0])):
                gen_text = vocab.sequence_to_text(generated[i].cpu().numpy())
                gt_text = vocab.sequence_to_text(targets[i].cpu().numpy())
                print(f"患者ID: {patient_ids[i]}")
                print(f"生成: {gen_text}")
                print(f"真实: {gt_text}")
                print("-" * 50)
            break


if __name__ == "__main__":
    main()