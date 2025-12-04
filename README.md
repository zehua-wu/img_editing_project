# Project Overview
This repository contains the implementation and experimental pipeline for our proposed DDIM + P2P editing framework enhanced with semantic control.

# Group Information
- **Group Number:** 02
- **Group Members:**
  - Weirong Chen
  - Ye Zhang 
  - Chengyi Zhou  

# Environment Setup
```shell
pip install -r environment/p2p_ctrl_requirement.txt
```

# Running Scripts
```shell
python run_editing_p2p.py 
    --edit_method_list ddim+p2p_ctrl
    --control_type 
    --control_scale 1.0
    --output_path ./output/
```

## Argument Details

| Argument | Description                                               |
|---------|-----------------------------------------------------------|
| `--edit_method_list` | Editing pipeline to use. For our method: `ddim+p2p_ctrl`. |
| `--control_type` | ControlNet type. Options: `canny`, `normal`, `depth`      |
| `--control_scale` | Strength of ControlNet guidance.                          |
| `--output_path` | Directory to save edited images.                          |
