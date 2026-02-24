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
from multiprocessing import Pool, cpu_count
from pathlib import Path
import re
from typing import List, Tuple, Optional


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


def collect_all_images(
    root_dirs: List[Optional[str | Path]],
    valid_extensions: tuple = ('.png', '.jpg', '.jpeg', '.JPG', '.PNG')
) -> List[Tuple[Optional[Path], Optional[Path], Optional[Path]]]:
    """
    递归遍历基准目录，收集图像路径三元组，支持部分模态缺失（路径为 None）
    """
    if not root_dirs or len(root_dirs) != 3:
        raise ValueError("root_dirs 必须是长度为3的列表")

    roots: List[Optional[Path]] = []
    for d in root_dirs:
        if d is None:
            roots.append(None)
        else:
            roots.append(Path(d).resolve())

    base_root = next((r for r in roots if r is not None), None)
    if base_root is None:
        raise ValueError("至少需要提供一个非 None 的根目录作为基准")

    print(f"基准目录: {base_root}")
    print(f"模态状态: mod1={roots[0] is not None}, mod2={roots[1] is not None}, mod3={roots[2] is not None}")

    triples: List[Tuple[Optional[Path], Optional[Path], Optional[Path]]] = []

    for root, _, files in os.walk(base_root):
        for file in files:
            if not file.lower().endswith(valid_extensions):
                continue

            p_base = Path(root) / file
            try:
                rel_path = p_base.relative_to(base_root)
            except ValueError:
                continue

            paths = [None, None, None]
            for idx, root_path in enumerate(roots):
                if root_path is not None:
                    candidate = root_path / rel_path
                    if candidate.is_file():
                        paths[idx] = candidate

            # 基准路径必须存在
            base_idx = roots.index(base_root)
            if paths[base_idx] is not None:
                triples.append(tuple(paths))  # type: ignore

    print(f"找到 {len(triples)} 组图像组合（允许部分模态缺失）")
    return triples


def extract_number_from_text(text: str) -> str:
    text = text.strip()
    match = re.search(r'<BOS>\s*([^<>\s]+)\s*<EOS>', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    fallback_match = re.search(r'([A-Za-z]{2,4}\d{4,8})', text)
    if fallback_match:
        return fallback_match.group(1).strip()

    return "未提取到有效编号"


def batch_generate_and_extract_number(
    triples: List[Tuple[Optional[Path], Optional[Path], Optional[Path]]],
    model,
    vocab: 'Vocabulary',
    transform,
    device,
    batch_size: int = 8,
    max_length: int = 64,
    temperature: float = 0.7,
    top_k: int = 40,
    output_base: str = "generated_numbers",           # 基础文件名，不带 .txt
    log_every: int = 20
):
    """
    批量推理，并按增强图像编号范围拆分成四个 txt 文件：
    - part1: aug_000001 ~ aug_000500
    - part2: aug_000501 ~ aug_001000
    - part3: aug_001001 ~ aug_001500
    - part4: aug_001501 ~ aug_002000
    """
    # ── 新增：自动创建输出目录 ──
    output_dir = os.path.dirname(output_base)
    if output_dir:  # 防止 output_base 没有目录部分
        os.makedirs(output_dir, exist_ok=True)
        print(f"已确保输出目录存在或创建：{output_dir}")
    else:
        print("输出路径没有目录部分，将直接在当前工作目录生成文件")

    model.eval()
    results = []

    # 定义四个输出文件（可根据需要修改命名）
    output_files = {
        1: f"{output_base}_part1.txt",   # 000001-00500
        2: f"{output_base}_part2.txt",   # 00501-01000
        3: f"{output_base}_part3.txt",   # 01001-01500
        4: f"{output_base}_part4.txt",   # 01501-02000
    }

    # 编号范围边界（闭区间）
    ranges = {
        1: (1, 1000),
        2: (1001, 2000),
        3: (2001, 3000),
        4: (3001, 4000),
    }

    # 打开所有文件句柄（追加模式，避免覆盖）
    file_handles = {}
    for part, fname in output_files.items():
        file_handles[part] = open(fname, 'w', encoding='utf-8')
        file_handles[part].write("图像相对路径\t生成文本\t提取的编号\t模态状态\n")

    dummy_tensor: Optional[torch.Tensor] = None
    global_idx = 0   # 用于跟踪当前处理的图像全局序号（从0开始）

    for batch_start in range(0, len(triples), batch_size):
        batch_end = min(batch_start + batch_size, len(triples))
        current_batch = triples[batch_start:batch_end]
        batch_len = len(current_batch)

        try:
            # 准备 batch 输入（与原代码相同）
            imgs1_list, imgs2_list, imgs3_list = [], [], []
            ref_paths = []
            status_list = []

            for p1_opt, p2_opt, p3_opt in current_batch:
                def load_or_zero(path_opt: Optional[Path]) -> torch.Tensor:
                    nonlocal dummy_tensor
                    if path_opt is not None:
                        img = Image.open(path_opt).convert('RGB')
                        return transform(img)
                    else:
                        if dummy_tensor is None:
                            dummy_tensor = torch.zeros(3, 512, 512, dtype=torch.float32)
                        return dummy_tensor.clone()

                imgs1_list.append(load_or_zero(p1_opt))
                imgs2_list.append(load_or_zero(p2_opt))
                imgs3_list.append(load_or_zero(p3_opt))

                ref_path = next((p for p in (p1_opt, p2_opt, p3_opt) if p is not None), None)
                ref_paths.append(ref_path)

                status_parts = []
                if p1_opt is None: status_parts.append("T2缺失")
                if p2_opt is None: status_parts.append("ADC缺失")
                if p3_opt is None: status_parts.append("DWI缺失")
                status_list.append(" / ".join(status_parts) if status_parts else "完整三模态")

            imgs1 = torch.stack(imgs1_list).to(device)
            imgs2 = torch.stack(imgs2_list).to(device)
            imgs3 = torch.stack(imgs3_list).to(device)

            with torch.no_grad(), torch.amp.autocast(
                device_type='cuda' if device.type == 'cuda' else 'cpu'
            ):
                gen_ids, _ = model.generate(
                    imgs1, imgs2, imgs3,
                    max_length=max_length,
                    start_token=vocab.word2idx.get('<BOS>', 1),
                    end_token=vocab.word2idx.get('<EOS>', 2),
                    temperature=temperature,
                    top_k=top_k,
                    use_greedy=True
                )

            gen_texts = [vocab.sequence_to_text(ids.cpu().numpy()) for ids in gen_ids]
            numbers = [extract_number_from_text(txt) for txt in gen_texts]

            # 写入 & 日志（按每个样本的“虚拟编号”分配文件）
            for j in range(batch_len):
                global_idx += 1   # 从 1 开始计数，对应 aug_000001 的 1
                rel_path = (
                    str(ref_paths[j].relative_to(ref_paths[j].parent.parent.parent))
                    if ref_paths[j] else "无有效路径"
                )
                gen_text = gen_texts[j]
                number = numbers[j]
                status = status_list[j]

                # 根据 global_idx 决定属于哪个 part
                part_num = None
                for p, (start, end) in ranges.items():
                    if start <= global_idx <= end:
                        part_num = p
                        break

                if part_num is not None:
                    line = f"{rel_path}\t{gen_text}\t{number}\t{status}\n"
                    file_handles[part_num].write(line)

                # 日志（保持原样）
                idx = batch_start + j
                if (idx + 1) % log_every == 0 or (idx + 1) == 1 or (idx + 1) == len(triples):
                    part_str = f"part{part_num}" if part_num else "未知part"
                    print(f"[{idx+1:4d}/{len(triples)}] {number:<12} ← {rel_path}  ({status})  → {part_str}")

                results.append((rel_path, gen_text, number, status))

        except Exception as batch_error:
            err_msg = f"Batch {batch_start+1}-{batch_end} 处理失败: {str(batch_error)}"
            print(err_msg)
            # 可以选择写入某个 error 文件，或忽略

    # 关闭所有文件
    for fh in file_handles.values():
        fh.close()

    print(f"\n已生成 4 个 txt 文件：")
    for part, fname in output_files.items():
        print(f"  - Part {part}: {fname}  ({ranges[part][0]:06d} ~ {ranges[part][1]:06d})")
    print(f"总计处理 {len(triples)} 组原始图像组合")
    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======================== 配置路径 ========================
    MOD1_DIR = r"G:\分析测试\仁济测试\T24000\X24-4156\L"       # T2
    MOD2_DIR = r"G:\分析测试\仁济测试\ADC4000\X24-4156\L"             # ADC
    MOD3_DIR = r"G:\分析测试\仁济测试\DWI4000\X24-4156\L"                         # DWI 故意缺失

    vocab_path = r"G:\分析测试\vocab.json"
    model_path = r"G:\分析测试\权重\model_多模态2-1月26.pth"

    # 加载词汇表和模型（保持原样）
    vocab = Vocabulary(min_freq=1)
    vocab.load(vocab_path)

    model = ImageToTextTransformer(
        vocab_size=vocab.vocab_size,
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        max_seq_length=11,
        image_feature_dim=2048,
        vocab=vocab
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    # 收集图像（支持 MOD3_DIR = None）
    image_triples = collect_all_images([MOD1_DIR, MOD2_DIR, MOD3_DIR])

    if not image_triples:
        print("没有找到任何图像组合")
    else:
        batch_generate_and_extract_number(
            image_triples,
            model,
            vocab,
            transform,
            device,
            batch_size=8,           # ← 新增：可根据显存调整 4/8/16/32 等
            max_length=11,
            temperature=0.75,
            top_k=35,
            output_base=r"G:\分析测试\匹配txt\verify\X24-4156\L4000\generated_numbers_2026",
            log_every=50
        )