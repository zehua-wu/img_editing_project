import os
import csv
import torch
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from models.p2p_editor import P2PEditor


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_latent_mask(org_mask, dest_size=(64, 64)):
    org_mask = (org_mask > 0).astype(np.uint8)

    # Resize via PIL (nearest preserves binary mask)
    pil_mask = Image.fromarray(org_mask * 255)
    pil_mask = pil_mask.resize(dest_size, Image.NEAREST)

    # Convert back to NumPy 0/1
    mask_np = np.array(pil_mask)
    mask_np = (mask_np > 127).astype(np.uint8)   # strict binary

    # Convert to torch tensor (1,1,H,W), still strictly 0/1
    mask_torch = torch.tensor(mask_np, dtype=torch.float16, device=device)
    # add batch + channel dims
    mask_torch = mask_torch.unsqueeze(0).unsqueeze(0)

    return mask_np, mask_torch

def mask_decode(encoded_mask, image_shape=[512,512]):
    length=image_shape[0]*image_shape[1]
    mask_array=np.zeros((length,))

    for i in range(0,len(encoded_mask),2):
        splice_len=min(encoded_mask[i+1],length-encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j]=1

    mask_array=mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0,:]=1
    mask_array[-1,:]=1
    mask_array[:,0]=1
    mask_array[:,-1]=1

    return mask_array


if __name__ == '__main__':
    data_dir = "PIE-Bench_v1"
    output_dir = "ddim+p2p_latent_mask"
    map_file_path = os.path.join(data_dir, "mapping_file.json")

    with open(map_file_path, "r") as f:
        annotation_file = json.load(f)

    edit_method = "ddim+p2p"
    p2p_editor = P2PEditor(
        method_list=[edit_method],
        device=device,
        num_ddim_steps=50
    )

    for key, item in tqdm(annotation_file.items()):
        mask = mask_decode(item["mask"])
        _, latent_mask = get_latent_mask(mask)

        setup_seed()
        torch.cuda.empty_cache()

        src_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        tgt_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
        image_path = os.path.join(f"{data_dir}/annotation_images", item["image_path"])

        edited_image = p2p_editor(
            edit_method,
            image_path=image_path,
            prompt_src=src_prompt,
            prompt_tar=tgt_prompt,
            guidance_scale=7.5,
            cross_replace_steps=0.4,
            self_replace_steps=0.6,
            latent_mask=latent_mask)

        output_path = os.path.join(f"{output_dir}/annotation_images", item["image_path"])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if isinstance(edited_image, Image.Image):
            edited_image.save(output_path)
        else:
            Image.fromarray(edited_image).save(output_path)
        # print("Saved to:", output_path)

        # show_image(edited_image)
        # break