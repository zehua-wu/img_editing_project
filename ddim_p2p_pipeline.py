# 第一部分：基础工具与图像处理
# 这部分包含了数学工具（如球面插值 Slerp）、图像加载与预处理、VAE 的编码解码工具，以及用于可视化的绘图工具。

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import PIL.Image as Image
import torch

from diffusers import StableDiffusionPipeline
try:
    from diffusers import ControlNetModel
except ImportError:
    ControlNetModel = None


# controlnet-aux 只在 control!=None 时才真正会用
from controlnet_aux import ZoeDetector, CannyDetector, OpenposeDetector



MAX_NUM_WORDS = 77
LATENT_SIZE = (64, 64)
LOW_RESOURCE = False

# --- 数学工具：球面线性插值 ---
def slerp(val, low, high):
    """ 
    Spherical Linear Interpolation (球面线性插值)
    用于在潜在空间中对两个向量进行平滑插值，比线性插值更能保持高维分布的特性。
    taken from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/4
    """
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res


def slerp_tensor(val, low, high):
    """ 
    对 Tensor 进行 Slerp 插值，常用于 Negative Prompt Inversion 或噪声混合。
    used in negtive prompt inversion
    """
    shape = low.shape
    res = slerp(val, low.flatten(1), high.flatten(1))
    return res.reshape(shape)

# --- 图像预处理 ---
def load_512(image_path, left=0, right=0, top=0, bottom=0):
    """
    加载图像并裁剪、缩放到 512x512 尺寸。
    """
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    # 简单的裁剪逻辑
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    # 中心裁剪为正方形
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    # 缩放到 512x512
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image

def preprocess_control_image(control_image, device):
    """
    将 controlnet_aux 输出的图转成 [-1,1] 的 tensor，形状 (B, C, H, W)
    """
    if isinstance(control_image, Image.Image):
        np_img = np.array(control_image.convert("RGB"))
    elif isinstance(control_image, np.ndarray):
        np_img = control_image
    else:
        raise ValueError("control_image 必须是 PIL.Image 或 numpy.ndarray")

    # 保证 512x512
    if np_img.shape[0] != 512 or np_img.shape[1] != 512:
        np_img = np.array(Image.fromarray(np_img).resize((512, 512)))

    np_img = np_img.astype(np.float32) / 255.0
    np_img = np_img.transpose(2, 0, 1)  # (C,H,W)
    np_img = 2.0 * np_img - 1.0        # [0,1] -> [-1,1]

    tensor = torch.from_numpy(np_img)[None].to(device)  # (1,C,H,W)
    return tensor

# --- VAE 与 Latent 交互 ---
def init_latent(latent, model, height, width, generator, batch_size):
    """
    初始化随机噪声（Latent），如果未提供则随机生成。
    """
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


@torch.no_grad()
def latent2image(model, latents, return_type='np'):
    """
    解码：将 Latent 空间向量通过 VAE Decoder 转换回像素图像。
    """
    latents = 1 / 0.18215 * latents.detach() # 0.18215 是 SD 的缩放因子
    image = model.decode(latents)['sample']
    if return_type == 'np':
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
    return image

@torch.no_grad()
def image2latent(model, image):
    """
    编码：将像素图像通过 VAE Encoder 转换为 Latent 向量。
    """
    with torch.no_grad():
        if type(image) is Image:
            image = np.array(image)
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            # 归一化到 [-1, 1]
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(model.device)
            latents = model.encode(image)['latent_dist'].mean
            latents = latents * 0.18215
    return latents

# --- 文本与可视化工具 ---

def get_word_inds(text: str, word_place: int, tokenizer):
    """
    获取特定单词在 Tokenizer 编码后的索引位置。
    用于确定 Prompt 中哪个词需要被替换或加强权重。
    """
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)

def update_alpha_time_word(alpha, bounds, prompt_ind,
                           word_inds=None):
    """
    更新时间步相关的权重矩阵 Alpha。
    用于控制在哪些时间步（Time Steps）进行 Attention 替换。
    """
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps,
                                   tokenizer, max_num_words=77):
    """
    生成控制 Attention 替换的时间步矩阵。
    """
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words

def txt_draw(text,
                target_size=[512,512]):
    """
    工具函数：将文本绘制成图片，用于最后拼接展示 Source Prompt 和 Target Prompt。
    """
    plt.figure(dpi=300,figsize=(1,1))
    plt.text(-0.1, 1.1, text,fontsize=3.5, wrap=True,verticalalignment="top",horizontalalignment="left")
    plt.axis('off')
    
    canvas = FigureCanvasAgg(plt.gcf())
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    # image = image.resize(target_size,Image.ANTIALIAS)
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    image = np.asarray(image)[:,:,:3]
    
    plt.close('all')
    
    return image






# 第二部分：Prompt 对齐算法
# 这是 P2P 的关键部分之一。为了在修改 Prompt 后（例如把 "cat" 改成 "dog"）还能复用原来的 Attention Map，必须知道新 Prompt 中的词与旧 Prompt 中的词的对应关系。
# 这里使用了 Needleman-Wunsch 算法（一种全局序列比对算法）来实现。


# %%
# Copyright 2022 Google LLC
# ... (License info skipped for brevity) ...
import torch
import numpy as np


class ScoreParams:
    """
    定义序列比对的打分参数：匹配给分，不匹配扣分，空缺扣分。
    """
    def __init__(self, gap, match, mismatch):
        self.gap = gap
        self.match = match
        self.mismatch = mismatch

    def mis_match_char(self, x, y):
        if x != y:
            return self.mismatch
        else:
            return self.match
        
    
def get_matrix(size_x, size_y, gap):
    """初始化比对矩阵"""
    matrix = []
    for i in range(len(size_x) + 1):
        sub_matrix = []
        for j in range(len(size_y) + 1):
            sub_matrix.append(0)
        matrix.append(sub_matrix)
    for j in range(1, len(size_y) + 1):
        matrix[0][j] = j*gap
    for i in range(1, len(size_x) + 1):
        matrix[i][0] = i*gap
    return matrix

# 注意：这里有一个同名函数 get_matrix，下面的版本使用了 numpy 优化，覆盖了上面的版本。
def get_matrix(size_x, size_y, gap):
    matrix = np.zeros((size_x + 1, size_y + 1), dtype=np.int32)
    matrix[0, 1:] = (np.arange(size_y) + 1) * gap
    matrix[1:, 0] = (np.arange(size_x) + 1) * gap
    return matrix


def get_traceback_matrix(size_x, size_y):
    """初始化回溯矩阵，用于记录路径"""
    matrix = np.zeros((size_x + 1, size_y +1), dtype=np.int32)
    matrix[0, 1:] = 1
    matrix[1:, 0] = 2
    matrix[0, 0] = 4
    return matrix


def global_align(x, y, score):
    """
    Needleman-Wunsch 全局比对算法实现。
    计算两个 Token 序列的最佳匹配路径。
    """
    matrix = get_matrix(len(x), len(y), score.gap)
    trace_back = get_traceback_matrix(len(x), len(y))
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            left = matrix[i, j - 1] + score.gap
            up = matrix[i - 1, j] + score.gap
            diag = matrix[i - 1, j - 1] + score.mis_match_char(x[i - 1], y[j - 1])
            matrix[i, j] = max(left, up, diag)
            if matrix[i, j] == left:
                trace_back[i, j] = 1
            elif matrix[i, j] == up:
                trace_back[i, j] = 2
            else:
                trace_back[i, j] = 3
    return matrix, trace_back


def get_aligned_sequences(x, y, trace_back):
    """根据回溯矩阵，生成对齐后的序列和映射关系"""
    x_seq = []
    y_seq = []
    i = len(x)
    j = len(y)
    mapper_y_to_x = []
    while i > 0 or j > 0:
        if trace_back[i, j] == 3: # 匹配或不匹配（对角线）
            x_seq.append(x[i-1])
            y_seq.append(y[j-1])
            i = i-1
            j = j-1
            mapper_y_to_x.append((j, i))
        elif trace_back[i][j] == 1: # 插入空缺（左移）
            x_seq.append('-')
            y_seq.append(y[j-1])
            j = j-1
            mapper_y_to_x.append((j, -1))
        elif trace_back[i][j] == 2: # 插入空缺（上移）
            x_seq.append(x[i-1])
            y_seq.append('-')
            i = i-1
        elif trace_back[i][j] == 4: # 结束
            break
    mapper_y_to_x.reverse()
    return x_seq, y_seq, torch.tensor(mapper_y_to_x, dtype=torch.int64)


def get_mapper(x, y, tokenizer, max_len=77):
    """
    对外接口：输入两个字符串，返回它们 Token 的映射关系 (Mapper)。
    用于 Refinement（改进）类型的编辑。
    """
    x_seq = tokenizer.encode(x)
    y_seq = tokenizer.encode(y)
    score = ScoreParams(0, 1, -1)
    matrix, trace_back = global_align(x_seq, y_seq, score)
    mapper_base = get_aligned_sequences(x_seq, y_seq, trace_back)[-1]
    alphas = torch.ones(max_len)
    alphas[: mapper_base.shape[0]] = mapper_base[:, 1].ne(-1).float()
    mapper = torch.zeros(max_len, dtype=torch.int64)
    mapper[:mapper_base.shape[0]] = mapper_base[:, 1]
    mapper[mapper_base.shape[0]:] = len(y_seq) + torch.arange(max_len - len(y_seq))
    return mapper, alphas


def get_refinement_mapper(prompts, tokenizer, max_len=77):
    """为所有 Target Prompts 生成相对于 Source Prompt 的映射"""
    x_seq = prompts[0]
    mappers, alphas = [], []
    for i in range(1, len(prompts)):
        mapper, alpha = get_mapper(x_seq, prompts[i], tokenizer, max_len)
        mappers.append(mapper)
        alphas.append(alpha)
    return torch.stack(mappers), torch.stack(alphas)


def get_word_inds(text, word_place, tokenizer):
    # (重复定义，已经在第一部分定义过，这里为了保持代码完整性保留)
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def get_replacement_mapper_(x, y, tokenizer, max_len=77):
    """
    用于 Replacement（替换）类型的编辑。
    手动计算替换词的 Attention Map 映射矩阵。
    """
    words_x = x.split(' ')
    words_y = y.split(' ')
    if len(words_x) != len(words_y):
        raise ValueError(f"attention replacement edit can only be applied on prompts with the same length"
                         f" but prompt A has {len(words_x)} words and prompt B has {len(words_y)} words.")
    inds_replace = [i for i in range(len(words_y)) if words_y[i] != words_x[i]]
    inds_source = [get_word_inds(x, i, tokenizer) for i in inds_replace]
    inds_target = [get_word_inds(y, i, tokenizer) for i in inds_replace]
    mapper = np.zeros((max_len, max_len))
    i = j = 0
    cur_inds = 0
    while i < max_len and j < max_len:
        if cur_inds < len(inds_source) and inds_source[cur_inds][0] == i:
            inds_source_, inds_target_ = inds_source[cur_inds], inds_target[cur_inds]
            if len(inds_source_) == len(inds_target_):
                mapper[inds_source_, inds_target_] = 1
            else:
                ratio = 1 / len(inds_target_)
                for i_t in inds_target_:
                    mapper[inds_source_, i_t] = ratio
            cur_inds += 1
            i += len(inds_source_)
            j += len(inds_target_)
        elif cur_inds < len(inds_source):
            mapper[i, j] = 1
            i += 1
            j += 1
        else:
            mapper[j, j] = 1
            i += 1
            j += 1

    return torch.from_numpy(mapper).float()



def get_replacement_mapper(prompts, tokenizer, max_len=77):
    x_seq = prompts[0]
    mappers = []
    for i in range(1, len(prompts)):
        mapper = get_replacement_mapper_(x_seq, prompts[i], tokenizer, max_len)
        mappers.append(mapper)
    return torch.stack(mappers)


# 第三部分：自定义 DDIM 调度器
# 这里重写了 DDIMScheduler 的 step 方法。
# 这是为了支持精确的反演（Inversion）和重构（Reconstruction），特别是处理方差（Variance）和噪声预测（pred_epsilon）的公式细节。
# %%
from typing import List, Optional, Tuple, Union
import torch
from diffusers import DDIMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler, DDIMSchedulerOutput
import torch.nn.functional as F
from torch.optim.adam import Adam


class DDIMSchedulerDev(DDIMScheduler):

    def step(
        self,
        model_output,
        timestep,
        sample,
        eta = 0.0,
        use_clipped_model_output = False,
        generator=None,
        variance_noise = None,
        return_dict = True,
        **kwargs,
    ):
        # 检查时间步是否设置
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # DDIM 核心公式实现 (参考论文 https://arxiv.org/pdf/2010.02502.pdf)
        # 符号映射: 
        # pred_noise_t -> e_theta(x_t, t) (模型预测的噪声)
        # pred_original_sample -> x_0 (预测的原图)

        # 1. 获取前一个时间步 (t-1)
        # prev_timestep = timestep_next
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 2. 计算 alpha, beta 参数
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. 计算预测的 x_0 (predicted original sample)
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            # predict V
            model_output = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )
        
        # 4. 裁剪预测的 x_0 (防止数值溢出)
        if kwargs.get("clip_sample", False):
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # 额外的重构引导逻辑 (Optional, 某些 P2P 变体使用)
        if kwargs.get("ref_image", None) is not None and kwargs.get("recon_lr", 0.0) > 0.0:
            ref_image = kwargs.get("ref_image").expand_as(pred_original_sample)
            recon_lr = kwargs.get("recon_lr", 0.0)
            recon_mask = kwargs.get("recon_mask", None)
            if recon_mask is not None:
                recon_mask = recon_mask.expand_as(pred_original_sample).float()
                pred_original_sample = pred_original_sample - recon_lr * (pred_original_sample - ref_image) * recon_mask
            else:
                pred_original_sample = pred_original_sample - recon_lr * (pred_original_sample - ref_image)

        # 5. 计算方差 sigma_t (当 eta > 0 时用于 DDPM 模式，否则为确定性 DDIM)
        if eta > 0:
            variance = self._get_variance(timestep, prev_timestep)
            std_dev_t = eta * variance ** (0.5)
        else:
            std_dev_t = 0.

        if use_clipped_model_output:
            # 如果 x_0 被裁剪，重新计算 model_output (噪声)
            model_output = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. 计算指向 x_t 的方向向量 (pred_sample_direction)
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output

        # 7. 计算前一步的样本 x_{t-1} (prev_sample)
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        # 如果有 eta，添加随机噪声 (转变为 DDPM 采样)
        if eta > 0:
            # randn_like does not support generator https://github.com/pytorch/pytorch/issues/27072
            device = model_output.device
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                if device.type == "mps":
                    # randn does not work reproducibly on mps
                    variance_noise = torch.randn(model_output.shape, dtype=model_output.dtype, generator=generator)
                    variance_noise = variance_noise.to(device)
                else:
                    variance_noise = torch.randn(
                        model_output.shape, generator=generator, device=device, dtype=model_output.dtype
                    )
            variance = self._get_variance(timestep, prev_timestep) ** (0.5) * eta * variance_noise

            prev_sample = prev_sample + variance

        if not return_dict:
            return (prev_sample,)

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)


# 第四部分：Attention 控制钩子
# 这一部分是 Prompt-to-Prompt 的魔法所在。
# 通过替换 UNet 中的 Cross-Attention 模块的 forward 函数，我们可以在 Attention Map 生成时对其进行拦截、保存、甚至修改（注入）。

# %%
# for diffuser newer version: padding
from diffusers.models.attention import CrossAttention
def _reshape_heads_to_batch_dim(self, tensor):
    """
    input: tensor [batch, seq_len, dim]
    output: [batch * heads, seq_len, dim // heads]
    """
    batch_size, seq_len, dim = tensor.shape
    head_dim = dim // self.heads
    tensor = tensor.view(batch_size, seq_len, self.heads, head_dim)
    tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * self.heads, seq_len, head_dim)
    return tensor

def _reshape_batch_dim_to_heads(self, tensor):
    """
    input: tensor [batch * heads, seq_len, head_dim]
    output: [batch, seq_len, heads * head_dim]
    """
    batch_size_times_heads, seq_len, head_dim = tensor.shape
    batch_size = batch_size_times_heads // self.heads
    tensor = tensor.view(batch_size, self.heads, seq_len, head_dim)
    tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.heads * head_dim)
    return tensor

# only works when CrossAttention fails then newer version applies
if not hasattr(CrossAttention, "reshape_heads_to_batch_dim"):
    CrossAttention.reshape_heads_to_batch_dim = _reshape_heads_to_batch_dim
if not hasattr(CrossAttention, "reshape_batch_dim_to_heads"):
    CrossAttention.reshape_batch_dim_to_heads = _reshape_batch_dim_to_heads


def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            **cross_attention_kwargs
        ):
            
            x = hidden_states

            # context：if encoder_hidden_states then cross-attn, else self-attn
            if isinstance(encoder_hidden_states, dict):  # !! here compatible 
                encoder_hidden_states = encoder_hidden_states.get("CONTEXT_TENSOR", None)

            if encoder_hidden_states is None:
                context = x
                is_cross = False
            else:
                context = encoder_hidden_states
                is_cross = True

            batch_size, sequence_length, dim = x.shape
            h = self.heads

            # Q from x || K/V from context 
            q = self.to_q(x)
            k = self.to_k(context)
            v = self.to_v(context)

            # reshape into multi-head 
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            # score
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            # attention_mask api
            if attention_mask is not None:
                # attention_mask usually [batch, 1, seq_len]
                max_neg_value = -torch.finfo(sim.dtype).max
                # into [batch*heads, seq_q, seq_k]
                # repeat to each head, too simple
                attn_mask = attention_mask.repeat(h, 1, 1)
                sim.masked_fill_(~attn_mask, max_neg_value)

            attn = sim.softmax(dim=-1)

            # P2P controller to edit
            # here MUST BE THIS!
            attn = controller(attn, is_cross, place_in_unet)

            out = torch.einsum("b i j, b j d -> b i d", attn, v)

            # back to batch*heads
            out = self.reshape_batch_dim_to_heads(out)

            # U-Net out
            return to_out(out)

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count
# 第五部分：Null-Text Inversion (反演)
# 这部分实现了 Null-Text Inversion。普通的反演只能近似回到原图，而 Null-Text Inversion 通过优化 unconditional embedding (空文本对应的 embedding)，
# 使得反演过程能精确重建原图。这是高质量编辑真实图像的前提。

# %%
class NullInversion:
    
    def prev_step(self, model_output, timestep: int, sample):
        """DDIM 采样反向步骤 (t -> t-1)"""
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output, timestep: int, sample):
        """DDIM 采样正向步骤 (Inversion, t -> t+1)"""
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        """单次噪声预测"""
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, guidance_scale, is_forward=True, context=None):
        """应用 Classifier-Free Guidance 的噪声预测"""
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else guidance_scale
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        """初始化 Prompt Embedding"""
        uncond_input = self.model.tokenizer(
            [""], 
            padding="max_length", 
            max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length", 
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        """执行标准的 DDIM Inversion 循环"""
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        """Inversion 入口函数"""
        latent = image2latent(self.model.vae, image)
        image_rec = latent2image(self.model.vae, latent)[0]
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon, guidance_scale):
        """
        Null-Text Optimization 核心逻辑。
        对每个时间步，优化 unconditional embedding，使得重建误差最小。
        """
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        for i in range(self.num_ddim_steps):
            uncond_embeddings = uncond_embeddings.clone().detach()
            t = self.model.scheduler.timesteps[i]
            if num_inner_steps!=0:
                uncond_embeddings.requires_grad = True
                # 针对 uncond_embeddings 进行优化
                optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
                latent_prev = latents[len(latents) - i - 2]
                with torch.no_grad():
                    noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
                for j in range(num_inner_steps):
                    noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                    loss = F.mse_loss(latents_prev_rec, latent_prev)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_item = loss.item()
                    # 提前停止条件
                    if loss_item < epsilon + i * 2e-5:
                        break
                
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, guidance_scale, False, context)
        return uncond_embeddings_list
    
    def invert(self, image_gt, prompt, guidance_scale, num_inner_steps=10, early_stop_epsilon=1e-5):
        """
        执行完整的 Inversion + Null-Text Optimization 流程。
        """
        self.init_prompt(prompt)
        register_attention_control(self.model, None)
        
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        
        # 优化空文本 Embedding
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon,guidance_scale)
        return image_gt, image_rec, ddim_latents, uncond_embeddings
    
    def __init__(self, model,num_ddim_steps):
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.prompt = None
        self.context = None
        self.num_ddim_steps=num_ddim_steps



# 第六部分：具体的 Attention 控制器类
# 这一部分定义了如何操作 Attention Map。

# AttentionStore: 仅保存，不修改（用于 Inversion 或获取原图 Attention）。

# AttentionReplace: 简单替换（用于 Replacement 编辑，如 "bike" -> "car"）。

# AttentionRefine: 精细替换（结合 Word Mapper，用于 Style 编辑或局部修正）。

# %%
import abc

# --- 基础抽象类 ---
class AttentionControl(abc.ABC):
    """Attention Controller 的基类"""
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross, place_in_unet):
        raise NotImplementedError

    def __call__(self, attn, is_cross, place_in_unet):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                # 仅处理 conditional 分支的 attention，保留 unconditional 的
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


# %%
def get_equalizer(text, word_select, values, tokenizer=None):
    """
    生成 Equalizer，用于增强或减弱特定词的 Attention 权重。
    """
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        inds = get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer

# %%
# (AttentionControl 类在此处重复定义了，为了保持源码完整性保留，逻辑同上)
class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross, place_in_unet):
        raise NotImplementedError

    def __call__(self, attn, is_cross, place_in_unet):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

# %%
# --- Attention 存储器 ---
class AttentionStore(AttentionControl):
    """
    用于存储 Attention Map，通常用于第一遍生成（Source Generation）。
    """

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross, place_in_unet):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead, 忽略过大的 feature map
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


# %%
# --- Attention 编辑基类 ---    
class AttentionControlEdit(AttentionStore, abc.ABC):
    """
    编辑模式的基类。它会根据当前时间步，决定是使用原图的 Attention 还是新生成的 Attention。
    """
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            # 将 Source 的 Self Attention 注入到 Target 中
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross, place_in_unet):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        # 如果处于需要替换的时间步范围内
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:] # attn_base 是 Source, attn_replace 是 Target
            if is_cross:
                # Cross Attention 替换：混合 alpha 权重
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                # Self Attention 替换：直接注入结构信息
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, 
                 prompts, 
                 num_steps,
                 cross_replace_steps,
                 self_replace_steps,
                 local_blend, 
                 tokenizer=None,
                 device="cuda"):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

# %%
# --- 具体的编辑实现 ---

class AttentionReplace(AttentionControlEdit):
    """
    Replacement 模式：使用 Mapper 矩阵将 Source 的 Attention 映射到 Target。
    适用于物体替换 (如 Cat -> Dog)。
    """

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps, cross_replace_steps, self_replace_steps,
                 local_blend = None, tokenizer=None,device="cuda"):
        super(AttentionReplace, self).__init__(prompts=prompts, 
                                                              num_steps=num_steps, 
                                                              cross_replace_steps=cross_replace_steps, 
                                                              self_replace_steps=self_replace_steps, 
                                                              local_blend=local_blend,
                                                              device=device)
        self.mapper = get_replacement_mapper(prompts, tokenizer).to(device)


class AttentionRefine(AttentionControlEdit):
    """
    Refinement 模式：用于更精细的编辑，使用之前计算的序列对齐 Mapper。
    """

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps, cross_replace_steps, self_replace_steps,
                 local_blend = None, tokenizer=None,device="cuda"):
        super(AttentionRefine, self).__init__(prompts=prompts, 
                                                              num_steps=num_steps, 
                                                              cross_replace_steps=cross_replace_steps, 
                                                              self_replace_steps=self_replace_steps, 
                                                              local_blend=local_blend,
                                                              device=device)
        self.mapper, alphas = get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])

class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, 
                 prompts, 
                 num_steps, 
                 cross_replace_steps, 
                 self_replace_steps, 
                 equalizer,
                 local_blend = None, 
                 controller = None,
                 device="cuda"):
        super(AttentionReweight, self).__init__(prompts=prompts, 
                                                num_steps=num_steps, 
                                                cross_replace_steps=cross_replace_steps, 
                                                self_replace_steps=self_replace_steps, 
                                                local_blend=local_blend,
                                                device=device)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller

# %%
class LocalBlend:
    """
    局部混合：通过特定词的 Attention Map 生成掩码 (Mask)，
    只在 Mask 区域内应用编辑，保留背景不受影响。
    """

    def get_mask(self, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = F.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = F.interpolate(maps, size=LATENT_SIZE)
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        mask = mask[:1] + mask
        return mask
    
    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:
            # 提取 Attention Map 来制作 Mask
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(maps, self.alpha_layers, True)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            # 混合 Latent：背景保持原样 (x_t[:1])，前景应用新生成 (x_t)
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

    def __init__(self, prompts, words, substruct_words=None, start_blend=0.2, th=(.3, .3),
                 tokenizer=None, device="cuda",num_ddim_steps=50):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        
        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * num_ddim_steps)
        self.counter = 0 
        self.th=th


# 第七部分：主流程与封装
# 最后是工厂函数（用于创建 Controller）和主执行管线。

# %%
def make_controller(pipeline, 
                    prompts, 
                    is_replace_controller, 
                    cross_replace_steps, 
                    self_replace_steps, 
                    blend_words=None, 
                    equilizer_params=None, 
                    num_ddim_steps=50,
                    device="cuda") -> AttentionControlEdit:
    """工厂函数：根据参数实例化正确的 Attention Controller"""
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_words, tokenizer=pipeline.tokenizer, device=device,num_ddim_steps=num_ddim_steps)
    if is_replace_controller:
        controller = AttentionReplace(prompts, 
                                      num_ddim_steps, 
                                      cross_replace_steps=cross_replace_steps, 
                                      self_replace_steps=self_replace_steps, 
                                      local_blend=lb,
                                      tokenizer=pipeline.tokenizer)
    else:
        controller = AttentionRefine(prompts, 
                                     num_ddim_steps, 
                                     cross_replace_steps=cross_replace_steps, 
                                     self_replace_steps=self_replace_steps, 
                                     local_blend=lb,
                                     tokenizer=pipeline.tokenizer)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], 
                           equilizer_params["words"], 
                           equilizer_params["values"], 
                           tokenizer=pipeline.tokenizer)
        controller = AttentionReweight(prompts, 
                                       num_ddim_steps,
                                       cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, 
                                       equalizer=eq, 
                                       local_blend=lb, 
                                       controller=controller)
    return controller

# %%
def p2p_guidance_diffusion_step(
    model,
    controller,
    latents,
    context,
    t,
    guidance_scale,
    low_resource=False,
    control_image=None,
    controlnet=None,
    controlnet_conditioning_scale: float = 1.0,
):
    if low_resource:
        ...
    else:
        latents_input = torch.cat([latents] * 2)

        if controlnet is not None and control_image is not None:
            if control_image.dim() == 3:
                control_image_ = control_image.unsqueeze(0)
            else:
                control_image_ = control_image

            if control_image_.shape[0] == 1 and latents_input.shape[0] > 1:
                control_image_ = control_image_.repeat(latents_input.shape[0], 1, 1, 1)

            # ① 先跑 ControlNet
            ctrl_out = controlnet(
                latents_input,
                t,
                encoder_hidden_states=context,
                controlnet_cond=control_image_,
                return_dict=True,
            )

            down_res = [r * controlnet_conditioning_scale for r in ctrl_out.down_block_res_samples]
            mid_res = ctrl_out.mid_block_res_sample * controlnet_conditioning_scale

            # ② 再把 residual 喂给 UNet
            unet_out = model.unet(
                latents_input,
                t,
                encoder_hidden_states=context,
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res,
            )
            if isinstance(unet_out, dict):
                noise_pred = unet_out["sample"]
            else:
                noise_pred = unet_out.sample
        else:
            # 原始无 ControlNet 分支
            noise_pred = model.unet(
                latents_input,
                t,
                encoder_hidden_states=context
            )["sample"]

        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)

    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_prediction_text - noise_pred_uncond
    )
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents




@torch.no_grad()
def p2p_guidance_forward(
    model,
    prompt,
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator = None,
    latent = None,
    uncond_embeddings=None,
    control_image=None,
    controlnet=None,
    controlnet_conditioning_scale: float = 1.0,
):
    """P2P 生成的前向循环，可选 ControlNet。"""
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]

    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)

    for i, t in enumerate(model.scheduler.timesteps):
        if uncond_embeddings_ is None:
            # Null-Text inversion 返回的是每一步的 uncond embedding 列表
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])

        latents = p2p_guidance_diffusion_step(
            model=model,
            controller=controller,
            latents=latents,
            context=context,
            t=t,
            guidance_scale=guidance_scale,
            low_resource=False,
            control_image=control_image,
            controlnet=controlnet,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )
        
    return latents, latent


# @torch.no_grad()
# def p2p_guidance_forward(
#     model,
#     prompt,
#     controller,
#     num_inference_steps: int = 50,
#     guidance_scale = 7.5,
#     generator = None,
#     latent = None,
#     uncond_embeddings=None
# ):
#     """P2P 生成的前向循环"""
#     batch_size = len(prompt)
#     register_attention_control(model, controller)
#     height = width = 512
    
#     text_input = model.tokenizer(
#         prompt,
#         padding="max_length",
#         max_length=model.tokenizer.model_max_length,
#         truncation=True,
#         return_tensors="pt",
#     )
#     text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
#     max_length = text_input.input_ids.shape[-1]
#     if uncond_embeddings is None:
#         uncond_input = model.tokenizer(
#             [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
#         )
#         uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
#     else:
#         uncond_embeddings_ = None

#     latent, latents = init_latent(latent, model, height, width, generator, batch_size)
#     model.scheduler.set_timesteps(num_inference_steps)
#     for i, t in enumerate(model.scheduler.timesteps):
#         if uncond_embeddings_ is None:
#             context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
#         else:
#             context = torch.cat([uncond_embeddings_, text_embeddings])
#         latents = p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
        
#     return latents, latent
class P2PDDIMEditor:
    """
    高层封装类：整合了模型加载、Inversion 和 Editing 的全流程。
    支持可选的 ControlNet: control in {None, "depth", "canny", "pose"}
    """
    def __init__(
        self,
        device,
        num_ddim_steps: int = 50,
        control: str = None,
        controlnet_conditioning_scale: float = 1.0,
    ) -> None:

        self.device = device
        self.num_ddim_steps = num_ddim_steps
        self.control = control
        self.controlnet_conditioning_scale = controlnet_conditioning_scale

        # 1) 初始化 DDIM 调度器（你自定义的版本）
        self.scheduler = DDIMSchedulerDev(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )

        # 2) 先创建基础 StableDiffusionPipeline（无 ControlNet）
        #    （注意：如果你用的是 sd1.5 的 controlnet，推荐这里用 runwayml/stable-diffusion-v1-5）
        self.ldm_stable = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            scheduler=self.scheduler,
        ).to(device)
        self.ldm_stable.scheduler.set_timesteps(self.num_ddim_steps)

        # 3) 再根据 control 类型，单独加载 ControlNetModel + 对应的 preprocessor
        self.controlnet = None
        self.preprocessor = None

        if control is not None:
            if ControlNetModel is None:
                raise ImportError("没有找到 ControlNetModel，请确认 diffusers 版本支持 ControlNet。")

            if control == "depth":
                controlnet_path = "lllyasviel/control_v11f1p_sd15_depth"
                self.preprocessor = ZoeDetector.from_pretrained("lllyasviel/Annotators")
            elif control == "canny":
                controlnet_path = "lllyasviel/control_v11p_sd15_canny"
                self.preprocessor = CannyDetector()
            elif control == "pose":
                controlnet_path = "lllyasviel/control_v11p_sd15_openpose"
                self.preprocessor = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            else:
                raise ValueError(f"Unknown control type: {control}")

            # 手动加载 ControlNetModel（不再用 ControlNetPipeline）
            self.controlnet = ControlNetModel.from_pretrained(controlnet_path).to(device)

    def __call__(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
    ):
        """
        直接执行 DDIM + P2P 编辑逻辑
        """
        return self.edit_image_ddim(
            image_path=image_path,
            prompt_src=prompt_src,
            prompt_tar=prompt_tar,
            guidance_scale=guidance_scale,
            cross_replace_steps=cross_replace_steps,
            self_replace_steps=self_replace_steps,
            blend_word=blend_word,
            eq_params=eq_params,
            is_replace_controller=is_replace_controller,
        )

    def edit_image_ddim(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
    ):
        # 1. 加载图片
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        # 2. 如果启用了 ControlNet，则基于原图生成 control_image tensor
        control_image_tensor = None
        if self.controlnet is not None and self.preprocessor is not None:
            pil_img_for_control = Image.fromarray(image_gt)
            control_img = self.preprocessor(pil_img_for_control)
            control_image_tensor = preprocess_control_image(control_img, self.device)

        # 3. 执行 DDIM Inversion（这里只用原始 UNet 反演，不带 ControlNet）
        null_inversion = NullInversion(
            model=self.ldm_stable,
            num_ddim_steps=self.num_ddim_steps,
        )

        _, _, x_stars, uncond_embeddings = null_inversion.invert(
            image_gt=image_gt,
            prompt=prompt_src,
            guidance_scale=guidance_scale,
            num_inner_steps=0,
        )
        x_t = x_stars[-1]

        # 4. 重建原图 (Reconstruction)
        controller = AttentionStore()
        reconstruct_latent, _ = p2p_guidance_forward(
            model=self.ldm_stable,
            prompt=[prompt_src],
            controller=controller,
            latent=x_t,
            num_inference_steps=self.num_ddim_steps,
            guidance_scale=guidance_scale,
            generator=None,
            uncond_embeddings=uncond_embeddings,
            control_image=control_image_tensor,           # 无 ControlNet 时就是 None
            controlnet=self.controlnet,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
        )

        reconstruct_image = latent2image(
            model=self.ldm_stable.vae, latents=reconstruct_latent
        )[0]

        # 文字说明
        image_instruct = txt_draw(
            f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}"
        )

        # 5. 执行编辑 (Edit)
        cross_replace_steps_dict = {"default_": cross_replace_steps}

        controller = make_controller(
            pipeline=self.ldm_stable,
            prompts=prompts,
            is_replace_controller=is_replace_controller,
            cross_replace_steps=cross_replace_steps_dict,
            self_replace_steps=self_replace_steps,
            blend_words=blend_word,
            equilizer_params=eq_params,
            num_ddim_steps=self.num_ddim_steps,
            device=self.device,
        )

        latents, _ = p2p_guidance_forward(
            model=self.ldm_stable,
            prompt=prompts,
            controller=controller,
            latent=x_t,  # 使用反演得到的噪声作为起点
            num_inference_steps=self.num_ddim_steps,
            guidance_scale=guidance_scale,
            generator=None,
            uncond_embeddings=uncond_embeddings,
            control_image=control_image_tensor,
            controlnet=self.controlnet,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
        )

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        # 6. 拼接结果: [文字说明, 原图, 重建图, 编辑图]
        return Image.fromarray(
            np.concatenate(
                (image_instruct, image_gt, reconstruct_image, images[-1]),
                axis=1,
            )
        )



# class P2PDDIMEditor:
#     """
#     高层封装类：整合了模型加载、Inversion 和 Editing 的全流程。
#     """
#     def __init__(self, device, num_ddim_steps=50) -> None:

#         self.device = device
#         self.num_ddim_steps = num_ddim_steps
        
#         # init model
#         # 初始化 DDIM 调度器
#         self.scheduler = DDIMSchedulerDev(beta_start=0.00085,
#                                         beta_end=0.012,
#                                         beta_schedule="scaled_linear",
#                                         clip_sample=False,
#                                         set_alpha_to_one=False)
        
#         # 加载 Stable Diffusion 模型
#         self.ldm_stable = StableDiffusionPipeline.from_pretrained(
#             "CompVis/stable-diffusion-v1-4", scheduler=self.scheduler).to(device)
#         self.ldm_stable.scheduler.set_timesteps(self.num_ddim_steps)



#     def __call__(self, 
#                 image_path,
#                 prompt_src,
#                 prompt_tar,
#                 guidance_scale=7.5,
#                 cross_replace_steps=0.4,
#                 self_replace_steps=0.6,
#                 blend_word=None,
#                 eq_params=None,
#                 is_replace_controller=False):
#         """
#         直接执行 DDIM + P2P 编辑逻辑
#         """
#         return self.edit_image_ddim(image_path, prompt_src, prompt_tar, 
#                                   guidance_scale=guidance_scale, 
#                                   cross_replace_steps=cross_replace_steps, 
#                                   self_replace_steps=self_replace_steps, 
#                                   blend_word=blend_word, 
#                                   eq_params=eq_params, 
#                                   is_replace_controller=is_replace_controller)

#     def edit_image_ddim(
#         self,
#         image_path,
#         prompt_src,
#         prompt_tar,
#         guidance_scale=7.5,
#         cross_replace_steps=0.4,
#         self_replace_steps=0.6,
#         blend_word=None,
#         eq_params=None,
#         is_replace_controller=False,
#     ):
#         # 1. 加载图片

#         image_gt = load_512(image_path)
#         prompts = [prompt_src, prompt_tar]

#         # 2. 执行 DDIM Inversion
#         # 原代码逻辑：使用 NullInversion 但设置 num_inner_steps=0，等同于标准 DDIM Inversion
#         null_inversion = NullInversion(model=self.ldm_stable,
#                                      num_ddim_steps=self.num_ddim_steps)
        
#         # invert 返回: (image_gt, image_enc, x_stars, uncond_embeddings)
#         # x_stars[-1] 是反演得到的噪声 (latents at t=T)
#         _, _, x_stars, uncond_embeddings = null_inversion.invert(
#             image_gt=image_gt, prompt=prompt_src, guidance_scale=guidance_scale, num_inner_steps=0)
#         x_t = x_stars[-1]

#         # 3. 重建原图 (Reconstruction) - 用于对比展示
#         # 这一步是为了生成 Source Prompt 对应的重建图
#         controller = AttentionStore()
#         reconstruct_latent, _ = p2p_guidance_forward(model=self.ldm_stable, 
#                                                     prompt=[prompt_src], 
#                                                     controller=controller, 
#                                                     latent=x_t, 
#                                                     num_inference_steps=self.num_ddim_steps, 
#                                                     guidance_scale=guidance_scale, 
#                                                     generator=None, 
#                                                     uncond_embeddings=uncond_embeddings)
        
#         reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]
        
#         # 生成文字说明图片
#         image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
#         # 4. 执行编辑 (Edit)
#         # 设置 P2P 的参数
#         cross_replace_steps_dict = {
#             'default_': cross_replace_steps,
#         }

#         # 创建 P2P 控制器
#         controller = make_controller(pipeline=self.ldm_stable,
#                                    prompts=prompts,
#                                    is_replace_controller=is_replace_controller,
#                                    cross_replace_steps=cross_replace_steps_dict,
#                                    self_replace_steps=self_replace_steps,
#                                    blend_words=blend_word,
#                                    equilizer_params=eq_params,
#                                    num_ddim_steps=self.num_ddim_steps,
#                                    device=self.device)
        
#         # 执行前向编辑过程
#         latents, _ = p2p_guidance_forward(model=self.ldm_stable, 
#                                         prompt=prompts, 
#                                         controller=controller, 
#                                         latent=x_t, # 使用反演得到的噪声作为起点
#                                         num_inference_steps=self.num_ddim_steps, 
#                                         guidance_scale=guidance_scale, 
#                                         generator=None, 
#                                         uncond_embeddings=uncond_embeddings)

#         # 将 latent 解码为图像
#         images = latent2image(model=self.ldm_stable.vae, latents=latents)

#         # 5. 拼接结果: [文字说明, 原图, 重建图, 编辑图]
#         return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image, images[-1]), axis=1))