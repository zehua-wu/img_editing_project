import os 
import numpy as np
import argparse
import json
from PIL import Image
import torch
import random

from models.p2p_editor import P2PEditor

def mask_decode(encoded_mask,image_shape=[512,512]):
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


def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


image_save_paths={
    "ddim+p2p":"ddim+p2p",
    "null-text-inversion+p2p":"null-text-inversion+p2p",
    "null-text-inversion+p2p_a800":"null-text-inversion+p2p_a800",
    "null-text-inversion+p2p_3090":"null-text-inversion+p2p_3090",
    "negative-prompt-inversion+p2p":"negative-prompt-inversion+p2p",
    "directinversion+p2p":"directinversion+p2p",
    "directinversion+p2p_guidance_0_1":"directinversion+p2p_guidance_0_1",
    "directinversion+p2p_guidance_0_5":"directinversion+p2p_guidance_0_5",
    "directinversion+p2p_guidance_0_25":"directinversion+p2p_guidance_0_25",
    "directinversion+p2p_guidance_0_75":"directinversion+p2p_guidance_0_75",
    "directinversion+p2p_guidance_1_1":"directinversion+p2p_guidance_1_1",
    "directinversion+p2p_guidance_1_5":"directinversion+p2p_guidance_1_5",
    "directinversion+p2p_guidance_1_25":"directinversion+p2p_guidance_1_25",
    "directinversion+p2p_guidance_1_75":"directinversion+p2p_guidance_1_75",
    "directinversion+p2p_guidance_25_1":"directinversion+p2p_guidance_25_1",
    "directinversion+p2p_guidance_25_5":"directinversion+p2p_guidance_25_5",
    "directinversion+p2p_guidance_25_25":"directinversion+p2p_guidance_25_25",
    "directinversion+p2p_guidance_25_75":"directinversion+p2p_guidance_25_75",
    "directinversion+p2p_guidance_5_1":"directinversion+p2p_guidance_5_1",
    "directinversion+p2p_guidance_5_5":"directinversion+p2p_guidance_5_5",
    "directinversion+p2p_guidance_5_25":"directinversion+p2p_guidance_5_25",
    "directinversion+p2p_guidance_5_75":"directinversion+p2p_guidance_5_75",
    "directinversion+p2p_guidance_75_1":"directinversion+p2p_guidance_75_1",
    "directinversion+p2p_guidance_75_5":"directinversion+p2p_guidance_75_5",
    "directinversion+p2p_guidance_75_25":"directinversion+p2p_guidance_75_25",
    "directinversion+p2p_guidance_75_75":"directinversion+p2p_guidance_75_75",
    "null-text-inversion+proximal-guidance":"null-text-inversion+proximal-guidance",
    "negative-prompt-inversion+proximal-guidance":"negative-prompt-inversion+proximal-guidance",
    "ablation_null-latent-inversion+p2p":"ablation_null-latent-inversion+p2p",
    "ablation_directinversion_08+p2p":"ablation_directinversion_08+p2p",
    "ablation_directinversion_04+p2p":"ablation_directinversion_04+p2p",
    "ablation_directinversion_interval_2+p2p":"ablation_directinversion_interval_2+p2p",
    "ablation_directinversion_interval_5+p2p":"ablation_directinversion_interval_5+p2p",
    "ablation_directinversion_interval_10+p2p":"ablation_directinversion_interval_10+p2p",
    "ablation_directinversion_interval_24+p2p":"ablation_directinversion_interval_24+p2p",
    "ablation_directinversion_interval_49+p2p":"ablation_directinversion_interval_49+p2p",
    "ablation_null-text-inversion_single_branch+p2p":"ablation_null-text-inversion_single_branch+p2p",
    "ablation_directinversion_add-source+p2p":"ablation_directinversion_add-source+p2p",
    "ablation_directinversion_add-target+p2p":"ablation_directinversion_add-target+p2p",

    # new
    "null-text-inversion+p2p_ctrl":"null-text-inversion+p2p_ctrl",
    "ddim+p2p_ctrl":"ddim+p2p_ctrl"
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action= "store_true") # rerun existing images
    parser.add_argument('--data_path', type=str, default="data") # the editing category that needed to run
    parser.add_argument('--output_path', type=str, default="output") # the editing category that needed to run
    parser.add_argument('--edit_category_list', nargs = '+', type=str, default=["0","1","2","3","4","5","6","7","8","9"]) # the editing category that needed to run
    parser.add_argument('--edit_method_list', nargs = '+', type=str, default=["ddim+p2p"]) # the editing methods that needed to run
    # add ctrlnet
    parser.add_argument("--control_type", type=str, choices=['canny','pose','depth','normal'],default=None,help="Control Type for ControlNet")
    parser.add_argument("--controlnet_path", type=str, default=None,
                    help="Path or HF id of ControlNet model")

    parser.add_argument("--control_scale", type=float, default=1.0,
                        help="Strength of ControlNet conditioning")
    args = parser.parse_args()
    
    rerun_exist_images=args.rerun_exist_images
    data_path=args.data_path
    output_path=args.output_path
    edit_category_list=args.edit_category_list
    edit_method_list=args.edit_method_list
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    p2p_editor = P2PEditor(
        method_list=edit_method_list,
        device=device,
        num_ddim_steps=50,
        control_type=args.control_type,
        controlnet_path=args.controlnet_path,
        control_scale=args.control_scale,
    ) 
    with open(f"{data_path}/mapping_file.json", "r") as f:
        editing_instruction = json.load(f)
    
    for key, item in editing_instruction.items():
        
        if item["editing_type_id"] not in edit_category_list:
            continue
        
        original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
        image_path = os.path.join(f"{data_path}/annotation_images", item["image_path"])
        editing_instruction = item["editing_instruction"]
        # blended_word = item["blended_word"].split(" ") if item["blended_word"] != "" else []
       
        # ===== SemanticControl Blended Word =======
        blended_raw = item.get("blended_word", "")
        if blended_raw != "":
            blended_tokens = blended_raw.split()
        else:
            blended_tokens = []

        #  prompts = [src, tar]
        #  words   = ((src_words...), (tar_words...))
        if len(blended_tokens) >= 2:
            # source: blended_tokens[0], target: blended_tokens[1]
            blend_word_struct = (
                (blended_tokens[0],),  # 对应 source prompt
                (blended_tokens[1],),  # 对应 target prompt
            )
            eq_params = {
                "words": (blended_tokens[1],),  # target token 提高权重（P2P equalizer β）
                "values": (2,),
            }
        else:
            blend_word_struct = None
            eq_params = None

       
        mask = Image.fromarray(np.uint8(mask_decode(item["mask"])[:,:,np.newaxis].repeat(3,2))).convert("L")

        for edit_method in edit_method_list:
            present_image_save_path=image_path.replace(data_path, os.path.join(output_path,image_save_paths[edit_method]))
            if args.control_type is not None:
                present_control_hint_save_path=image_path.replace(data_path, os.path.join(output_path,image_save_paths[edit_method],'control_hint',args.control_type))
                present_semantic_alpha_vis_hint_save_path=image_path.replace(data_path, os.path.join(output_path,image_save_paths[edit_method],'semantic_alpha',args.control_type))
            if ((not os.path.exists(present_image_save_path)) or rerun_exist_images):
                print(f"editing image [{image_path}] with [{edit_method}]")
                setup_seed()
                torch.cuda.empty_cache()

                if args.control_type == None:
                    edited_image = p2p_editor(edit_method,
                                            image_path=image_path,
                                            prompt_src=original_prompt,
                                            prompt_tar=editing_prompt,
                                            guidance_scale=7.5,
                                            cross_replace_steps=0.4,
                                            self_replace_steps=0.6,
                                            blend_word= blend_word_struct,
                                            eq_params= eq_params,
                                            proximal="l0",
                                            quantile=0.75,
                                            use_inversion_guidance=True,
                                            recon_lr=1,
                                            recon_t=400,
                                            )
                    if not os.path.exists(os.path.dirname(present_image_save_path)):
                        os.makedirs(os.path.dirname(present_image_save_path))
                    edited_image.save(present_image_save_path)
                    
                    print(f"finish")
                
                else:
                    edited_image, control_hint,semantic_alpha_vis = p2p_editor(edit_method,
                        image_path=image_path,
                        prompt_src=original_prompt,
                        prompt_tar=editing_prompt,
                        guidance_scale=7.5,
                        cross_replace_steps=0.4,
                        self_replace_steps=0.6,
                        blend_word= blend_word_struct,
                        eq_params= eq_params,
                        proximal="l0",
                        quantile=0.75,
                        use_inversion_guidance=True,
                        recon_lr=1,
                        recon_t=400,
                        )
                    if not os.path.exists(os.path.dirname(present_image_save_path)):
                        os.makedirs(os.path.dirname(present_image_save_path))
                    edited_image.save(present_image_save_path)

                    if not os.path.exists(os.path.dirname(present_control_hint_save_path)):
                        os.makedirs(os.path.dirname(present_control_hint_save_path))
                    control_hint.save(present_control_hint_save_path)

                    if not os.path.exists(os.path.dirname(present_semantic_alpha_vis_hint_save_path)):
                        os.makedirs(os.path.dirname(present_semantic_alpha_vis_hint_save_path))
                    semantic_alpha_vis.save(present_semantic_alpha_vis_hint_save_path)
                    
                    print(f"finish")

                
            else:
                print(f"skip image [{image_path}] with [{edit_method}]")
        
        