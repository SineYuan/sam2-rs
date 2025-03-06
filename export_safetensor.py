
import os.path as osp

import yaml
import torch
from safetensors.torch import save_file

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def flatten_dict(d, parent_key='', sep='.'):
    items = {}
    for k, v in d.items():
        if k == '_target_':
            continue  # 忽略_target_键
        
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            # 过滤字符串和None值
            if isinstance(v, str) or v is None:
                continue
            
            # 转换值为Tensor
            if isinstance(v, bool):
                tensor = torch.tensor(v, dtype=torch.uint8)
                #tensor = torch.tensor(v, dtype=torch.bool)
            elif isinstance(v, (int, float)):
                tensor = torch.tensor(v)
            elif isinstance(v, list):
                try:
                    tensor = torch.tensor(v)
                except TypeError:  # 处理包含非数值类型的列表
                    continue
            else:
                continue  # 忽略其他类型
            
            items[new_key] = tensor
    return items


checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
# 读取YAML文件
model_cfg_path = osp.join('sam2', model_cfg)
with open(model_cfg_path, "r") as f:
    yaml_data = yaml.safe_load(f)

yaml_data['config'] = yaml_data.pop('model')
# 展平字典并转换值为Tensor
config_dict = flatten_dict(yaml_data)

# 保存为safetensors文件
save_file(config_dict, "config.safetensors")

sam_model = build_sam2(model_cfg, checkpoint, device="cpu")

image_size = sam_model.image_size
input_tensor = torch.ones((1, 3, image_size, image_size), dtype=torch.float32)
patch_embed = sam_model.image_encoder.trunk.patch_embed(input_tensor)
# precompute pos_embed and save to safetensors file
img_size_pos_embed = sam_model.image_encoder.trunk._get_pos_embed(patch_embed.shape[1:3])

del sam_model

state_dict = torch.load(checkpoint, map_location="cpu")

state_dict = state_dict.pop('model')
state_dict.update(config_dict)

state_dict['image_encoder.trunk.img_size_pos_embed'] = img_size_pos_embed.cpu().contiguous()

safetensors_file = osp.splitext(checkpoint)[0] + ".safetensors"
save_file(state_dict, safetensors_file)
print(f"Converted {checkpoint} to {safetensors_file}")

if __name__ == "__main__":
    pass
