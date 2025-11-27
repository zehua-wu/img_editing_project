import os
import numpy as np
import argparse
import json
from PIL import Image
import torch
import random
from typing import Optional, List, Tuple

# 假设您提供的 P2P 代码文件名为 ddim_p2p_pipeline.py
# 导入您提供的代码中的 P2PDDIMEditor 类
from ddim_p2p_pipeline import P2PDDIMEditor, MAX_NUM_WORDS

# --- 辅助函数 ---

def mask_decode(encoded_mask: List[int], image_shape: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    解码 RLE 编码的 Mask。
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
            count = encoded_mask[idx+1]
        except IndexError:
            break
            
        splice_len = min(count, length - start)
        if start < length and splice_len > 0:
            mask_array[start:start + splice_len] = 1
        idx += 2
            
    mask_array = mask_array.reshape(height, width)
    
    # to avoid annotation errors in boundary
    mask_array[0,:]=1
    mask_array[-1,:]=1
    mask_array[:,0]=1
    mask_array[:,-1]=1
            
    return mask_array


def setup_seed(seed=1234):
    """设置随机种子以确保结果可复现"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- 映射表 ---
# 仅保留用于文件路径的映射，但请注意，P2PDDIMEditor 不会根据这些键切换编辑逻辑。
image_save_paths={
    "ddim+p2p":"ddim_p2p",
    "null-text-inversion+p2p":"nti_p2p",
    "directinversion+p2p":"directinversion_p2p",
    # ... 其他路径根据您的实际需求保留或删除 ...
    }


# --- 主执行逻辑 ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action= "store_true") # rerun existing images
    parser.add_argument('--data_path', type=str, default="data") # the editing category that needed to run
    parser.add_argument('--output_path', type=str, default="output") # the editing category that needed to run
    parser.add_argument('--edit_category_list', nargs = '+', type=str, default=["0","1","2","3","4","5","6","7","8","9"]) # the editing category that needed to run
    parser.add_argument('--edit_method_list', nargs = '+', type=str, default=["ddim+p2p"]) # the editing methods that needed to run
    args = parser.parse_args()
    
    rerun_exist_images = args.rerun_exist_images
    data_path = args.data_path
    output_path = args.output_path
    edit_category_list = args.edit_category_list
    edit_method_list = args.edit_method_list
    
    # 实例化您提供的 P2PDDIMEditor
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_ddim_steps = 50
    # 类的名称已从 P2PEditor 更改为 P2PDDIMEditor
    p2p_editor = P2PDDIMEditor(device=device, num_ddim_steps=num_ddim_steps)
    
    print(f"Loaded P2PDDIMEditor on {device} with {num_ddim_steps} steps.")
    
    # 确保保存路径映射表中只使用存在的键
    valid_edit_method_list = [m for m in edit_method_list if m in image_save_paths]
    if not valid_edit_method_list:
        print("Warning: None of the requested edit methods are in image_save_paths. Using default path.")
        
    with open(f"{data_path}/mapping_file.json", "r") as f:
        editing_instruction = json.load(f)
    
    for key, item in editing_instruction.items():
        
        if item["editing_type_id"] not in edit_category_list:
            continue
        
        original_prompt = item["original_prompt"].replace("[", "").replace("]", "").strip()
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "").strip()
        image_path = os.path.join(f"{data_path}/annotation_images", item["image_path"])
        # editing_instruction = item["editing_instruction"] # 未使用
        
        # 处理 blended_word 列表
        blended_word_raw = item["blended_word"].split(" ") if item["blended_word"] else []
        
        # P2P 的 blend_word 参数结构：需要 (Source word tuple, Target word tuple)
        # 这里假设 blended_word_raw 中包含的是 source 和 target 对应的单个词
        if len(blended_word_raw) >= 2:
             blend_word_param = (blended_word_raw[0], blended_word_raw[1])
             # 确保 LocalBlend 的 words 参数格式正确：
             # LocalBlend 需要一个 list of words for EACH prompt
             blend_word_param = [ (blended_word_raw[0],), (blended_word_raw[1],) ]
        else:
             blend_word_param = None
        
        # 假设 Replacement/Refinement 的选择逻辑：
        # 如果 prompt 词数不同，通常用 Refinement (Mapper)
        # 如果 prompt 词数相同 (如 cat -> dog)，通常用 Replacement (Mapper)
        # 这里我们保守地使用一个简单的启发式：
        is_replace_controller = len(original_prompt.split()) == len(editing_prompt.split())
        
        # Mask 只是被解码了，但在您的 P2PDDIMEditor 中没有被使用。
        # 如果需要 Local Blend，它会通过 Attention Map 自动计算 Mask。
        # mask = Image.fromarray(np.uint8(mask_decode(item["mask"])[:,:,np.newaxis].repeat(3,2))).convert("L")

        for edit_method in edit_method_list:
            
            if edit_method not in image_save_paths:
                print(f"Skip edit method [{edit_method}]: Path not defined.")
                continue

            present_image_save_path = image_path.replace(data_path, os.path.join(output_path, image_save_paths[edit_method]))
            
            if not os.path.exists(os.path.dirname(present_image_save_path)):
                os.makedirs(os.path.dirname(present_image_save_path))
            
            if ((not os.path.exists(present_image_save_path)) or rerun_exist_images):
                print(f"Editing image [{image_path}] with [{edit_method}]")
                setup_seed()
                torch.cuda.empty_cache()
                
                # --- 核心调用修改 ---
                # 您的 P2PDDIMEditor 不接受 edit_method, proximal, quantile, use_inversion_guidance, recon_lr, recon_t 等参数
                
                # 简化 Equalizer 参数，仅在 blend_word 存在时启用
                eq_params = None
                if blend_word_param and len(blended_word_raw) >= 2:
                     # 假设 Equalizer 只对 Target Prompt 的 blended_word[1] 施加权重 2
                     eq_params = {
                         "words": (blended_word_raw[1], ),
                         "values": (2.0, ) # 使用浮点数
                     }
                
                edited_image = p2p_editor(
                    image_path=image_path,
                    prompt_src=original_prompt,
                    prompt_tar=editing_prompt,
                    guidance_scale=7.5,
                    cross_replace_steps=0.4,
                    self_replace_steps=0.6,
                    # blend_word 必须是 (Source word, Target word) 的形式，否则 LocalBlend 初始化会失败
                    blend_word=blend_word_param, 
                    eq_params=eq_params,
                    is_replace_controller=is_replace_controller # 默认替换模式
                )
                
                edited_image.save(present_image_save_path)
                print(f"Finish and saved to {present_image_save_path}")
            else:
                print(f"Skip image [{image_path}] with [{edit_method}] (already exists)")
