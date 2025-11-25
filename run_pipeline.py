import os
import numpy as np
import argparse
import json
from PIL import Image
import torch
import random
from typing import List, Tuple

# 假设您提供的 P2P 代码文件名为 ddim_p2p_pipeline.py
# 导入您提供的代码中的 P2PDDIMEditor 类
from ddim_p2p_pipeline import P2PDDIMEditor, MAX_NUM_WORDS


# --- 辅助函数 ---

def mask_decode(encoded_mask: List[int], image_shape: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    解码 RLE 编码的 Mask。
    目前脚本里没有直接用到 mask，只是保留以便以后做基于 mask 的编辑。
    """
    height, width = image_shape
    length = height * width
    mask_array = np.zeros((length,), dtype=np.uint8)

    # 确保编码长度是偶数
    if len(encoded_mask) % 2 != 0:
        encoded_mask = encoded_mask[:-1]

    idx = 0
    while idx < len(encoded_mask):
        try:
            start = encoded_mask[idx]
            count = encoded_mask[idx + 1]
        except IndexError:
            break

        splice_len = min(count, length - start)
        if start < length and splice_len > 0:
            mask_array[start:start + splice_len] = 1
        idx += 2

    mask_array = mask_array.reshape(height, width)

    # to avoid annotation errors in boundary
    mask_array[0, :] = 1
    mask_array[-1, :] = 1
    mask_array[:, 0] = 1
    mask_array[:, -1] = 1

    return mask_array


def setup_seed(seed: int = 1234):
    """设置随机种子以确保结果可复现"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- 映射表 ---
# 仅保留用于文件路径的映射，
# P2PDDIMEditor 本身不会根据这些 key 切换编辑逻辑。
image_save_paths = {
    "ddim+p2p": "ddim_p2p",
    "null-text-inversion+p2p": "nti_p2p",
    "directinversion+p2p": "directinversion_p2p",
    # ... 其他路径根据您的实际需求保留或删除 ...
}


# --- 主执行逻辑 ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--rerun_exist_images',
        action="store_true",
        help="如果已存在输出图像，是否重新生成"
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default="data",
        help="数据根目录，里面需要有 mapping_file.json 和 annotation_images"
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default="output",
        help="输出根目录"
    )
    parser.add_argument(
        '--edit_category_list',
        nargs='+',
        type=str,
        default=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        help="要运行的 editing_type_id 列表"
    )
    parser.add_argument(
        '--edit_method_list',
        nargs='+',
        type=str,
        default=["ddim+p2p"],
        help="编辑方法列表，只影响输出子目录命名"
    )

    # ========= 新增：ControlNet 相关参数 =========
    parser.add_argument(
        '--control',
        type=str,
        default="none",
        choices=["none", "depth", "canny", "pose"],
        help="ControlNet type: none / depth / canny / pose"
    )
    parser.add_argument(
        '--control_scale',
        type=float,
        default=1.0,
        help="ControlNet conditioning scale"
    )
    # ==========================================

    args = parser.parse_args()

    rerun_exist_images = args.rerun_exist_images
    data_path = args.data_path
    output_path = args.output_path
    edit_category_list = args.edit_category_list
    edit_method_list = args.edit_method_list
    control_type = args.control
    control_scale = args.control_scale

    # 设备与步数
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_ddim_steps = 50

    # 这里假设你已经把 P2PDDIMEditor 改成了：
    # P2PDDIMEditor(
    #   device,
    #   num_ddim_steps=50,
    #   control=None/"depth"/"canny"/"pose",
    #   controlnet_conditioning_scale=1.0
    # )
    p2p_editor = P2PDDIMEditor(
        device=device,
        num_ddim_steps=num_ddim_steps,
        control=None if control_type == "none" else control_type,
        controlnet_conditioning_scale=control_scale,
    )

    print(
        f"Loaded P2PDDIMEditor on {device} with {num_ddim_steps} steps. "
        f"control={control_type}, control_scale={control_scale}"
    )

    # 只保留有定义输出路径的编辑方法
    valid_edit_method_list = [m for m in edit_method_list if m in image_save_paths]
    if not valid_edit_method_list:
        print("Warning: None of the requested edit methods are in image_save_paths. "
              "No images will be saved under named subfolders.")

    mapping_file_path = os.path.join(data_path, "mapping_file.json")
    with open(mapping_file_path, "r") as f:
        editing_instruction = json.load(f)

    for key, item in editing_instruction.items():

        # 1) 过滤类别
        if item["editing_type_id"] not in edit_category_list:
            continue

        # 2) 提取 prompt & 图像路径
        original_prompt = item["original_prompt"].replace("[", "").replace("]", "").strip()
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "").strip()
        image_path = os.path.join(data_path, "annotation_images", item["image_path"])

        # 3) blended_word 处理
        blended_word_raw = item["blended_word"].split(" ") if item["blended_word"] else []

        # LocalBlend(words) 期望一个 list，长度 == len(prompts)
        # 每个元素是对应 prompt 的词列表或单个词。
        # 这里构造为：
        #   prompts = [prompt_src, prompt_tar]
        #   blend_word = [ (src_word,), (tgt_word,) ]
        if len(blended_word_raw) >= 2:
            blend_word_param = [
                (blended_word_raw[0],),  # src prompt 里要 blend 的词
                (blended_word_raw[1],),  # tar prompt 里要 blend 的词
            ]
        else:
            blend_word_param = None

        # 4) 选择 Replacement 或 Refinement：
        #    这里沿用你原来的启发式：词数一致 -> Replacement，否则 Refinement
        is_replace_controller = len(original_prompt.split()) == len(editing_prompt.split())

        for edit_method in valid_edit_method_list:

            # 决定保存路径：用 edit_method 对应的子目录
            present_image_save_path = image_path.replace(
                data_path,
                os.path.join(output_path, image_save_paths[edit_method])
            )

            os.makedirs(os.path.dirname(present_image_save_path), exist_ok=True)

            if (not os.path.exists(present_image_save_path)) or rerun_exist_images:
                print(f"Editing image [{image_path}] with [{edit_method}], control={control_type}")
                setup_seed()
                torch.cuda.empty_cache()

                # 5) Equalizer 参数（可选）
                #    简单策略：如果有 blended_word，则对 target prompt 中的 blended_word[1] 放大权重
                eq_params = None
                if blend_word_param and len(blended_word_raw) >= 2:
                    eq_params = {
                        "words": (blended_word_raw[1],),  # target prompt 中的词
                        "values": (2.0,),                 # 放大到 2 倍
                    }

                # 6) 核心调用：适配新版 P2PDDIMEditor 接口
                edited_image = p2p_editor(
                    image_path=image_path,
                    prompt_src=original_prompt,
                    prompt_tar=editing_prompt,
                    guidance_scale=7.5,
                    cross_replace_steps=0.4,
                    self_replace_steps=0.6,
                    blend_word=blend_word_param,
                    eq_params=eq_params,
                    is_replace_controller=is_replace_controller,
                )

                edited_image.save(present_image_save_path)
                print(f"Finish and saved to {present_image_save_path}")
            else:
                print(f"Skip image [{image_path}] with [{edit_method}] (already exists)")
