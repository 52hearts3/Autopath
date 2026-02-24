import random
import numpy as np
import math
from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from PIL import Image
from kornia.losses import SSIMLoss
import torchvision.transforms as transforms

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子
set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def norm_layer(channels):
    return nn.GroupNorm(32, channels)


class DynamicConditionProjection(nn.Module):
    """动态条件投影门控模块"""

    def __init__(self, cond_dim=256, time_dim=256, hidden_dim=512,temp=0.5):
        super().__init__()
        self.temp = temp  # 新增温度参数
        self.gate_net = nn.Sequential(
            nn.Linear(cond_dim + time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=1)
        )
        self.proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3 * time_dim)
        )

    def forward(self, time_emb, cond):
        # 输入形状检查
        assert time_emb.dim() == 2 and cond.dim() == 2, "输入必须是2D张量"
        # 动态权重计算
        gate_logits = self.gate_net[:-1](torch.cat([time_emb, cond], dim=1))
        gate_weights = F.softmax(gate_logits / self.temp + 1e-6, dim=1)  # 应用温度缩放

        # 条件特征变换
        proj_params = self.proj(cond).view(-1, 3, time_emb.size(1))  # [B, 3, D_t]

        # 加权融合
        weighted = (proj_params * gate_weights.unsqueeze(-1)).sum(dim=1)  # [B, D_t]

        # 残差连接
        return time_emb + weighted


class AdaGN(nn.Module):
    """条件自适应组归一化"""

    def __init__(self, cond_dim=256, num_channels=128):
        super().__init__()
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, 4 * num_channels),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.utils.weight_norm(nn.Linear(4 * num_channels, 2 * num_channels))
        )
        self.norm = nn.GroupNorm(32, num_channels)

    def forward(self, x, cond):
        # 输入形状检查
        assert x.dim() == 4, "输入特征图必须是4D张量"
        assert cond.dim() == 2, "条件向量必须是2D张量"

        # 生成缩放和偏移参数
        scale, shift = self.cond_encoder(cond).chunk(2, dim=1)  # 各为[B, C]

        # 应用自适应归一化
        x = self.norm(x)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(stride=2)

    def forward(self, x):
        return self.op(x)


class CrossAttention(nn.Module):
    def __init__(self, channels, context_dim=384, heads=4, dim_head=64):  # 修改context_dim默认值
        super().__init__()
        self.channels = channels
        self.context_dim = context_dim
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.norm = norm_layer(channels)
        self.to_q = nn.Conv2d(channels, inner_dim, kernel_size=1, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)  # 输入维度必须等于context_dim
        self.to_out = nn.Conv2d(inner_dim, channels, kernel_size=1)

    def forward(self, x, context):
        b, c, h, w = x.shape

        # 确保context的维度匹配
        assert context.size(1) == self.context_dim, f"条件维度错误，期望{self.context_dim}，实际得到{context.size(1)}"

        # 其余代码保持不变
        x_norm = self.norm(x)
        q = self.to_q(x_norm)
        q = q.view(b, self.heads, self.dim_head, -1).permute(0, 1, 3, 2)

        kv = self.to_kv(context).view(b, -1, 2, self.heads, self.dim_head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        scale = 1 / math.sqrt(self.dim_head)
        attn = torch.einsum('b h i d, b h j d -> b h i j', q * scale, k)
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.permute(0, 1, 3, 2).reshape(b, -1, h, w)
        return self.to_out(out) + x

class TimestepBlock(nn.Module):


    @abstractmethod
    def forward(self, x, t):
        """
        Apply the models to `x` given `t` timestep embeddings.
        """
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, t, cond):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, t, cond)
            elif isinstance(layer, CrossAttention):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x


class ResidualBlock(TimestepBlock):
    """改进的残差块"""

    def __init__(self, in_channels, out_channels, time_channels, cond_dim=384, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 第一个卷积块（使用AdaGN）
        self.norm1 = AdaGN(cond_dim, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # 时间/条件处理（保持原结构）
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
        self.cond_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_channels)
        )

        # 第二个卷积块（使用AdaGN）
        self.norm2 = AdaGN(cond_dim, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # 快捷连接
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t, cond):
        # 条件自适应归一化
        h = self.conv1(self.act1(self.norm1(x, cond)))  # 传入cond

        # 时间+条件嵌入
        h += self.time_emb(t)[:, :, None, None] + self.cond_emb(cond)[:, :, None, None]

        # 第二层处理
        h = self.conv2(self.dropout(self.act2(self.norm2(h, cond))))  # 再次传入cond

        return h + self.shortcut(x)


class ControlNet(nn.Module):
    def __init__(self, in_channels, model_channels, cond_dim, num_res_blocks=2,
                 attention_resolutions=(8, 16), dropout=0., channel_mult=(1, 2, 2),
                 conv_resample=True, num_heads=8):
        super().__init__()
        self.model_channels = model_channels
        time_embed_dim = model_channels * 4

        # 时间嵌入和条件投影
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.utils.spectral_norm(nn.Linear(time_embed_dim, time_embed_dim)),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, time_embed_dim),
            nn.LayerNorm(time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # 初始化下采样模块和零卷积层
        self.down_blocks = nn.ModuleList()
        self.zero_convs = nn.ModuleList()  # 修改为动态创建

        # 初始卷积层
        ch = model_channels
        self.down_blocks.append(nn.Conv2d(in_channels, ch, 3, padding=1))
        self.zero_convs.append(self._make_zero_conv(ch))  # 初始层的零卷积

        # 构建下采样层
        ds = 1
        current_res = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                # 残差块和注意力
                layers = [
                    ResidualBlock(
                        in_channels=ch,
                        out_channels=mult * model_channels,
                        time_channels=time_embed_dim,
                        cond_dim=time_embed_dim,
                        dropout=dropout
                    )
                ]
                ch = mult * model_channels
                if current_res in attention_resolutions:
                    layers.append(CrossAttention(ch, context_dim=time_embed_dim, heads=num_heads))

                # 添加模块并创建对应的零卷积
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self._make_zero_conv(ch))

            # 下采样层（非最后一级时）
            if level != len(channel_mult) - 1:
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                self.zero_convs.append(self._make_zero_conv(ch))
                ds *= 2
                current_res = ds

    def _make_zero_conv(self, channels):
        conv = nn.Conv2d(channels, channels, kernel_size=1)
        # 微小随机初始化（替代全零初始化）
        nn.init.normal_(conv.weight, mean=0, std=1e-6)
        nn.init.zeros_(conv.bias)
        return conv

    def forward(self, x, t_emb, cond):
        t = self.time_embed(t_emb)
        cond_proj = self.cond_proj(cond)

        features = []
        h = x
        zero_idx = 0

        for module in self.down_blocks:
            # 主路径前向传播
            if isinstance(module, TimestepEmbedSequential):
                h = module(h, t, cond_proj)
            else:
                h = module(h)

            # 应用对应零卷积（每个模块后都应用）
            h = h + self.zero_convs[zero_idx](h)
            if isinstance(module, (ResidualBlock, TimestepEmbedSequential)) and any(
                    isinstance(layer, ResidualBlock) for layer in module.children()):
                features.append(h)  # 只在残差块后保存特征
            zero_idx += 1

        return features

class UNetModel(nn.Module):
    def __init__(
            self,
            in_channels=3,
            model_channels=96,
            out_channels=3,
            cond_dim=256,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            dropout=0.,
            channel_mult=(1, 2, 2),
            conv_resample=True,
            num_heads=8,
            use_controlnet=False
    ):
        super().__init__()

        self.model_channels = model_channels

        # ==================== 维度配置 ====================
        time_embed_dim = model_channels * 4  # 384 when model_channels=96

        if use_controlnet:
            self.use_controlnet = ControlNet(
                in_channels=3,  # 假设控制图像是HE图像
                model_channels=model_channels,
                cond_dim=time_embed_dim,
                num_res_blocks=num_res_blocks,
                channel_mult=channel_mult
            )

        # ==================== 条件投影 ====================
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, time_embed_dim),
            nn.utils.spectral_norm(nn.Linear(time_embed_dim, time_embed_dim)),  # 谱归一化
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # ==================== 时间嵌入 ====================
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # ==================== 动态门控融合 ====================
        self.dynamic_proj = DynamicConditionProjection(
            cond_dim=time_embed_dim,  # 使用投影后的维度
            time_dim=time_embed_dim,
            hidden_dim=512
        )

        # ==================== 下采样模块 ====================
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        self.down_blocks.append(TimestepEmbedSequential(
            nn.Conv2d(in_channels, ch, 3, padding=1)
        ))
        down_block_chans = [ch]

        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(
                        in_channels=ch,
                        out_channels=mult * model_channels,
                        time_channels=time_embed_dim,
                        cond_dim=time_embed_dim,  # 使用投影后的维度
                        dropout=dropout
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(CrossAttention(
                        channels=ch,
                        context_dim=time_embed_dim,  # 关键修改
                        heads=num_heads
                    ))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_chans.append(ch)
                ds *= 2

        # ==================== 中间模块 ====================
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(
                ch, ch,
                time_channels=time_embed_dim,
                cond_dim=time_embed_dim,
                dropout=dropout
            ),
            CrossAttention(
                channels=ch,
                context_dim=time_embed_dim,  # 关键修改
                heads=num_heads
            ),
            ResidualBlock(
                ch, ch,
                time_channels=time_embed_dim,
                cond_dim=time_embed_dim,
                dropout=dropout
            )
        )

        # ==================== 上采样模块 ====================
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(
                        in_channels=ch + down_block_chans.pop(),
                        out_channels=model_channels * mult,
                        time_channels=time_embed_dim,
                        cond_dim=time_embed_dim,
                        dropout=dropout
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(CrossAttention(
                        channels=ch,
                        context_dim=time_embed_dim,  # 关键修改
                        heads=num_heads
                    ))
                if level > 0 and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        # ==================== 输出层 ====================
        self.out = nn.Sequential(
            AdaGN(cond_dim=time_embed_dim, num_channels=ch),  # 使用投影后的维度
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )

    def forward(self, x, timesteps, cond, control_img=None):  # 新增control_img参数
        # 输入验证
        # assert cond.shape[1] == 256, f"条件输入维度应为256，实际得到{cond.shape[1]}"
        if self.use_controlnet:
            assert control_img is not None, "启用ControlNet时必须提供控制图像"
            assert control_img.shape[2:] == x.shape[2:], "控制图像尺寸必须与输入一致"

        # ========== 基础处理 ==========
        # 时间嵌入
        t_emb = timestep_embedding(timesteps, self.model_channels)
        t = self.time_embed(t_emb)

        # 条件投影
        cond_proj = self.cond_proj(cond)

        # 动态融合
        t = self.dynamic_proj(t, cond_proj)

        # ========== ControlNet处理 ==========
        control_features = []
        if self.use_controlnet:
            # 提取控制特征（带梯度）
            with torch.enable_grad():
                control_features = self.use_controlnet(
                    control_img,  # (B, 3, H, W)
                    t_emb,  # 时间嵌入
                    cond_proj  # 投影后的条件
                )

        # ========== 下采样过程 ==========
        hs = []
        h = x
        ctrl_idx = 0  # 控制特征索引

        for i, module in enumerate(self.down_blocks):
            # 主路径前向传播
            h = module(h, t, cond_proj)

            # 特征融合（仅在残差块后融合）
            if self.use_controlnet and isinstance(module[0], ResidualBlock):
                # 获取对应层控制特征
                ctrl_feat = control_features[ctrl_idx]
                # 特征相加（带可学习缩放系数）
                h = h + 0.1 * ctrl_feat  # 初始缩放系数设为0.1
                ctrl_idx += 1

            hs.append(h)

        # ========== 中间块 ==========
        h = self.middle_block(h, t, cond_proj)

        # ========== 上采样过程 ==========
        for module in self.up_blocks:
            # 跳跃连接
            h = torch.cat([h, hs.pop()], dim=1)

            # 主路径前向传播
            h = module(h, t, cond_proj)

        # ========== 输出处理 ==========
        h = self.out[0](h, cond_proj)  # AdaGN
        h = self.out[1](h)  # SiLU
        h = self.out[2](h)  # 最终卷积
        return h

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

ssim_loss_fn = SSIMLoss(window_size=7).to(device)  # 窗口大小适配 256x256 图像

# SSIM 损失函数
def ssim_loss(pred, target):
    pred = pred.clamp(-1, 1)
    target = target.clamp(-1, 1)
    return ssim_loss_fn(pred, target)

class GaussianDiffusion:
    def __init__(
        self,
        timesteps=1000,
        beta_schedule='linear'
    ):
        self.timesteps = timesteps

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def _extract(self, a: torch.FloatTensor, t: torch.LongTensor, x_shape):
        # get the param of given timestep t
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    def q_sample(self, x_start: torch.FloatTensor, t: torch.LongTensor, noise=None):
        # forward diffusion (using the nice property): q(x_t | x_0)
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def q_mean_variance(self, x_start: torch.FloatTensor, t: torch.LongTensor):
        # Get the mean and variance of q(x_t | x_0).
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_start: torch.FloatTensor, x_t: torch.FloatTensor, t: torch.LongTensor):
        # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t: torch.FloatTensor, t: torch.LongTensor, noise: torch.FloatTensor):
        # compute x_0 from x_t and pred noise: the reverse of `q_sample`
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_mean_variance(self, model, x_t, t, cond, he_control ,clip_denoised=True):  # 新增cond参数
        pred_noise = model(x_t, t, cond,control_img=he_control )  # 传入条件
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, model, x_t, t, cond, he_control,clip_denoised=True):  # 新增cond参数
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t, cond, he_control,clip_denoised)
        noise = torch.randn_like(x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img

    @torch.no_grad()
    def sample(self, model, cond, control_img, image_size, batch_size=8, channels=3):  # 新增control_img参数
        shape = (batch_size, channels, image_size, image_size)
        device = next(model.parameters()).device

        # 验证控制图像尺寸
        assert control_img.shape[-2:] == (image_size, image_size), \
            f"控制图像尺寸需为{image_size}x{image_size}, 当前为{control_img.shape[-2:]}"

        img = torch.randn(shape, device=device)

        # 处理条件维度
        if cond.dim() == 1:
            cond = cond.unsqueeze(-1).repeat(1, 256)
        cond = cond.to(device).float()

        # 处理控制图像批次维度
        if control_img.size(0) != batch_size:
            control_img = control_img[:1].repeat(batch_size, 1, 1, 1)

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(
                model,
                img,
                t,
                cond,
                control_img  # 传递控制图像
            )
        return img

    @torch.no_grad()
    def ddim_sample(self, model, x_t, t, t_prev, cond, he_control, eta=0.0, clip_denoised=True):
        """
        DDIM 单步采样
        eta: 0.0 为确定性 DDIM，1.0 为 DDPM-like 随机，推荐 0.0~1.0
        """
        pred_noise = model(x_t, t, cond, control_img=he_control)
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)  # 预测 x_0

        if clip_denoised:
            x_recon = torch.clamp(x_recon, -1., 1.)

        # DDIM 公式中的方向项 (指向 x_prev 的均值)
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_alpha_cumprod_prev = self._extract(self.sqrt_alphas_cumprod, t_prev, x_t.shape)

        direction = sqrt_alpha_cumprod_prev * x_recon

        # sigma: 控制随机噪声的强度
        sigma = eta * torch.sqrt(
            (1 - sqrt_alpha_cumprod_prev ** 2) / (1 - sqrt_alpha_cumprod_t ** 2) *
            (1 - sqrt_alpha_cumprod_t ** 2 / sqrt_alpha_cumprod_prev ** 2)
        )

        noise = torch.randn_like(x_t) if eta > 0 else 0.0

        x_prev = direction + torch.sqrt(1 - sqrt_alpha_cumprod_prev ** 2 - sigma ** 2) * pred_noise + sigma * noise

        return x_prev

    @torch.no_grad()
    def sample_ddim(self, model, cond, control_img, image_size, batch_size=8, channels=3,
                    ddim_steps=50, eta=0.0, schedule_type='uniform'):
        """
        DDIM 完整采样循环
        ddim_steps: 采样步数（推荐 50~200，越少越快）
        eta: 随机性参数（推荐 0.0 以获得最稳定结果）
        schedule_type: 'uniform'（均匀）或 'quadratic'（二次，更好质量）
        """
        shape = (batch_size, channels, image_size, image_size)
        device = next(model.parameters()).device

        # 验证控制图像尺寸（同原代码）
        assert control_img.shape[-2:] == (image_size, image_size), \
            f"控制图像尺寸需为{image_size}x{image_size}, 当前为{control_img.shape[-2:]}"

        # 处理 cond 和 control_img（同原 sample）
        if cond.dim() == 1:
            cond = cond.unsqueeze(-1).repeat(1, 256)
        cond = cond.to(device).float()

        if control_img.size(0) != batch_size:
            control_img = control_img[:1].repeat(batch_size, 1, 1, 1)

        # 生成 DDIM 时间序列
        if schedule_type == 'uniform':
            indices = torch.linspace(self.timesteps - 1, 0, ddim_steps + 1).long()  # 包含 0
        elif schedule_type == 'quadratic':
            indices = ((torch.linspace(0, math.sqrt(self.timesteps * 0.8), ddim_steps + 1)) ** 2).long()
        else:
            raise ValueError("schedule_type 仅支持 'uniform' 或 'quadratic'")

        times = indices[:-1].to(device)  # 当前 t (从大到小)
        times_prev = indices[1:].to(device)  # 下一个 t_prev

        # 从纯噪声开始
        img = torch.randn(shape, device=device)

        for i in tqdm(range(ddim_steps), desc=f'DDIM sampling ({ddim_steps} steps)'):
            t = torch.full((batch_size,), times[i], device=device, dtype=torch.long)
            t_prev = torch.full((batch_size,), times_prev[i], device=device, dtype=torch.long)

            img = self.ddim_sample(
                model, img, t, t_prev, cond, control_img, eta=eta
            )

        return img

    def train_losses(self, model, x_start, t, cond, control_img, ssim_weight=0.1):
        """
        计算训练损失，返回总损失、MSE 损失和 SSIM 损失。

        参数：
            model: 扩散模型
            x_start: 原始图像 [B, C, H, W]
            t: 时间步 [B]
            cond: 条件向量 [B, cond_dim]
            control_img: 控制图像 [B, C, H, W]
            ssim_weight: SSIM 损失权重

        返回：
            dict: 包含 total_loss, mse_loss, ssim_loss
        """
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = model(x_noisy, t, cond, control_img=control_img)

        # MSE 损失
        mse_loss = F.mse_loss(noise, predicted_noise)

        # 恢复 x_0
        x_recon = self.predict_start_from_noise(x_noisy, t, predicted_noise)
        x_recon = torch.clamp(x_recon, -1, 1)

        # SSIM 损失
        ssim_loss_val = ssim_loss(x_recon, x_start)

        # 总损失
        total_loss = mse_loss + ssim_weight * ssim_loss_val

        return {
            "total_loss": total_loss,
            "mse_loss": mse_loss,
            "ssim_loss": ssim_loss_val
        }


def adaptive_noise(cond, min_scale=0.05, max_scale=0.3):
    """基于条件向量幅度的自适应噪声"""
    cond_norm = torch.norm(cond, dim=1, keepdim=True)
    scale = min_scale + (max_scale - min_scale) * (cond_norm / cond_norm.max())
    return torch.randn_like(cond) * scale

batch_size = 2
timesteps = 1000

# define model and diffusion
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNetModel(
    in_channels=3,
    model_channels=96,  # 提升基础通道数以增强特征表达能力
    out_channels=3,
    channel_mult=(1, 2, 4),  # 增加下采样深度至4级
    attention_resolutions=[8, 16],  # 扩展注意力到更多分辨率层级
    num_heads=8,  # 保持多头注意力机制平衡
    cond_dim=512,
    dropout=0.1,  # 适当降低防止过拟合
    num_res_blocks=3,  # 增加残差块密度
    conv_resample=True,
    use_controlnet=True
)

model.to(device)
gaussian_diffusion = GaussianDiffusion(timesteps=timesteps,beta_schedule='cosine')


class MRI2HEDataset(Dataset):
    def __init__(self, mri_dir, he_dir, transform=None):
        super().__init__()
        self.mri_dir = mri_dir
        self.he_dir = he_dir
        self.transform = transform

        # 获取 MRI 和 HE 文件名（去掉扩展名）并匹配
        mri_files = {os.path.splitext(f)[0]: f for f in os.listdir(mri_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))}
        he_files = {os.path.splitext(f)[0]: f for f in os.listdir(he_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))}

        # 找到共同的文件名（交集）
        common_names = sorted(set(mri_files.keys()) & set(he_files.keys()))
        if not common_names:
            raise ValueError("没有找到匹配的 MRI 和 HE 图像文件名！")

        # 构建配对的文件列表
        self.mri_images = [mri_files[name] for name in common_names]
        self.he_images = [he_files[name] for name in common_names]

        # 检查文件数量是否一致
        if len(self.mri_images) != len(self.he_images):
            raise ValueError("MRI 和 HE 图像数量不一致！")

        # 初始化索引列表用于洗牌
        self.indices = list(range(len(self.mri_images)))

    def shuffle_indices(self):
        """在每个 epoch 开始时洗牌索引"""
        random.shuffle(self.indices)

    def __len__(self):
        return len(self.mri_images)

    def __getitem__(self, idx):
        # 使用洗牌后的索引
        idx = self.indices[idx]
        mri_path = os.path.join(self.mri_dir, self.mri_images[idx])
        he_path = os.path.join(self.he_dir, self.he_images[idx])

        mri_image = Image.open(mri_path).convert("RGB")
        he_image = Image.open(he_path).convert("RGB")

        # 同步数据增强
        if self.transform:
            seed = torch.randint(0, 100000, (1,)).item()
            torch.manual_seed(seed)
            mri_image = self.transform(mri_image)
            torch.manual_seed(seed)
            he_image = self.transform(he_image)

        # 对 HE 图像进行归一化
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        he_image = normalize(he_image)
        he_image = torch.clamp(he_image, -1.0, 1.0)

        return {"mask": mri_image, "he": he_image}

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.RandomResizedCrop(
        size=512,
        scale=(0.8, 1.0),
        ratio=(0.9, 1.1),
        interpolation=transforms.InterpolationMode.NEAREST
    ),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor()
])

# 加载数据集
dataset = MRI2HEDataset(
    mri_dir=r"/media/lenovo/6ED3FFE79A41910F/CT-HE测试/dataset/ddpm/mask",
    he_dir=r"/media/lenovo/6ED3FFE79A41910F/CT-HE测试/dataset/ddpm/he",
    transform=transform
)

# 创建 DataLoader，设置实际 batch_size=4（通过梯度累积模拟 batch_size=32）
batch_size = 2
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

model.load_state_dict(torch.load(r'he_batch345-3.pth'))
model.to(device)

# 生成并可视化结果
model.eval()

### ADDED: 直方图匹配
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity

def apply_histogram_matching(generated, reference):
    """
    对生成图像应用直方图匹配
    参数：
        generated: 生成图像数组 (n, H, W, C)
        reference: 参考图像数组 (n, H, W, C)
    返回：
        匹配后的图像数组
    """
    matched_images = np.zeros_like(generated)
    for i in range(len(generated)):
        matched_images[i] = match_histograms(
            generated[i],
            reference[i],
            channel_axis=-1  # 重要！指定通道维度
        )
    return matched_images



# 初始化特征提取模型
def create_feature_extractor(device):
    # 加载预训练EfficientNet_B0
    effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    # 移除分类层，保留特征提取器
    effnet.classifier = torch.nn.Identity()  # 输出特征维度1280
    effnet.eval()
    return effnet.to(device)


# 特征提取函数
def extract_features(img_array, model, device):
    from torchvision.transforms import functional as F
    """从numpy图像提取特征"""
    # 转换图像格式并预处理
    img_tensor = F.to_tensor(img_array).unsqueeze(0).to(device)  # [1, 3, H, W]
    img_tensor = F.normalize(img_tensor,
                             mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    # 提取特征
    with torch.no_grad():
        features = model(img_tensor)
    return features.squeeze().cpu().numpy()


# 可视化辅助函数
def plot_images(original, generated, matched, diff, similarity, ax_row):
    """绘制单行图像结果"""
    ax_row[0].imshow(original)
    ax_row[0].set_title('Original', fontsize=8)
    ax_row[0].axis('off')

    ax_row[1].imshow(generated)
    ax_row[1].set_title('Generated', fontsize=8)
    ax_row[1].axis('off')

    ax_row[2].imshow(matched)
    ax_row[2].set_title('Matched', fontsize=8)
    ax_row[2].axis('off')

    ax_row[3].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
    ax_row[3].set_title(f'Diff Map\nMean: {diff.mean():.3f}', fontsize=8)
    ax_row[3].axis('off')

    ax_row[4].text(0.5, 0.5,
                   f"Feature Similarity:\n{similarity:.3f}",
                   ha='center', va='center',
                   fontsize=10, color='blue')
    ax_row[4].axis('off')


def process_batch(batch_data, model, cond_model, diffusion, feature_extractor, device):
    test_mri, test_he = batch_data['mask'], batch_data['he']
    test_he = test_he.to(device)
    test_mri = test_mri.to(device)

    # ---------- 生成条件 ----------
    with torch.no_grad():
        # 动态 batch_size
        batch_size = test_he.size(0)
        cond = torch.zeros(batch_size, 512)

    # ---------- 扩散模型生成 ----------
    generated_images = diffusion.sample_ddim(
        model=model,
        cond=cond,
        control_img=test_mri,
        image_size=test_he.shape[2],
        batch_size=batch_size,
        channels=3,
        ddim_steps=100,  # ← 根據需求調整 25/50/100
        eta=0.5,  # 確定性最好
        schedule_type='uniform'  # 或試試 'quadratic'
    )

    # ---------- 生成图像 → numpy ----------
    generated_images = generated_images.detach().cpu().numpy()
    generated_images = np.transpose(generated_images, (0, 2, 3, 1))   # [B, H, W, C]
    generated_images = (generated_images + 1) / 2
    generated_images = np.clip(generated_images, 0, 1)

    # ---------- 真实 HE 图像 → numpy ----------
    original_images = test_he.detach().cpu().numpy()
    original_images = np.transpose(original_images, (0, 2, 3, 1))
    original_images = (original_images + 1) / 2
    original_images = np.clip(original_images, 0, 1)

    # ---------- 直方图匹配 ----------
    matched_images = apply_histogram_matching(generated_images, original_images)

    # ==============================
    #   保存：生成图像 + 真实图像
    # ==============================
    save_dir = r'/media/lenovo/6ED3FFE79A41910F/CT-HE测试/dataset/img'
    os.makedirs(save_dir, exist_ok=True)

    # 推断当前 batch 编号（防止覆盖）
    # 每张图像对应两个文件（gen + real），所以除以 2*当前 batch_size
    existing_files = len(os.listdir(save_dir))
    batch_idx = existing_files // (2 * batch_size)

    for i in range(batch_size):
        # 生成图像
        gen_path = os.path.join(save_dir,
                                f"gen_batch{batch_idx:04d}_img{i:02d}.png")
        Image.fromarray((generated_images[i] * 255).astype(np.uint8)).save(gen_path)

        # 真实图像
        real_path = os.path.join(save_dir,
                                 f"real_batch{batch_idx:04d}_img{i:02d}.png")
        Image.fromarray((original_images[i] * 255).astype(np.uint8)).save(real_path)

    print(f"Batch {batch_idx}: "
          f"saved {batch_size} generated + {batch_size} real images → {save_dir}")

    # 返回值保持不变（后续可视化仍可使用）
    return original_images, generated_images, matched_images

def visualize_batch(batch_idx, original, generated, matched, feature_extractor, device):
    # 计算特征相似度
    def calc_sim(orig, gen):
        return [cosine_similarity(
            [extract_features(o, feature_extractor, device)],
            [extract_features(g, feature_extractor, device)]
        )[0][0] for o, g in zip(orig, gen)]

    orig_sim = calc_sim(original, generated)
    matched_sim = calc_sim(original, matched)

    # 创建可视化，设置 squeeze=False 确保 axes 始终为二维数组
    batch_size = len(original)
    fig, axes = plt.subplots(batch_size, 5, figsize=(15, 3 * batch_size), squeeze=False)

    for i in range(batch_size):
        # 计算差异图
        diff = np.abs(original[i] - matched[i]).mean(axis=-1)

        # 绘制图像，统一使用 axes[i] 作为当前行的子图数组
        plot_images(
            original=original[i],
            generated=generated[i],
            matched=matched[i],
            diff=diff,
            similarity=matched_sim[i],
            ax_row=axes[i]  # 直接传递当前行的子图数组
        )
        # 在第五个子图添加原始相似度文本
        axes[i, 4].text(0.5, 0.25,
                        f"(Original Sim: {orig_sim[i]:.3f})",
                        ha='center', va='center',
                        fontsize=8, color='gray')

    plt.suptitle(f"Batch {batch_idx} Generation Results", y=0.98)
    plt.tight_layout()
    plt.show()


# 主函数
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_extractor = create_feature_extractor(device)

    # 控制处理的批次数量
    max_batches_to_show = 300  # 根据需要调整这个值
    start = 0
    for batch_idx, batch_data in enumerate(dataloader):
        if batch_idx >= max_batches_to_show:
            break
        if batch_idx <= start:
            continue
        print(f"\nProcessing batch {batch_idx}:")

        # 处理批次
        orig, gen, matched = process_batch(
            batch_data=batch_data,
            model=model,
            cond_model=None,
            diffusion=gaussian_diffusion,
            feature_extractor=feature_extractor,
            device=device
        )

        # 可视化结果
        visualize_batch(
            batch_idx=batch_idx,
            original=orig,
            generated=gen,
            matched=matched,
            feature_extractor=feature_extractor,
            device=device
        )

main()