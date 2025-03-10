import os.path as osp
import argparse
import yaml
import torch
from safetensors.torch import save_file

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def flatten_dict(d, parent_key='', sep='.'):
    items = {}
    for k, v in d.items():
        if k == '_target_':
            continue
        
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            if isinstance(v, str) or v is None:
                continue
            
            if isinstance(v, bool):
                tensor = torch.tensor(v, dtype=torch.uint8)
            elif isinstance(v, (int, float)):
                tensor = torch.tensor(v)
            elif isinstance(v, list):
                try:
                    tensor = torch.tensor(v)
                except TypeError:
                    continue
            else:
                continue
            
            items[new_key] = tensor
    return items

def main():
    parser = argparse.ArgumentParser(description="Convert SAM2 checkpoint to safetensors format")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to the input checkpoint file")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the model configuration YAML file")
    parser.add_argument("--output", type=str, 
                        help="Path to the output safetensors file (default: based on checkpoint name)")
    
    args = parser.parse_args()

    # Process configuration
    model_cfg_path = osp.join('sam2', args.config)
    with open(model_cfg_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    yaml_data['config'] = yaml_data.pop('model')
    config_dict = flatten_dict(yaml_data)

    # Build model and process pos_embed
    sam_model = build_sam2(args.config, args.checkpoint, device="cpu")
    image_size = sam_model.image_size
    input_tensor = torch.ones((1, 3, image_size, image_size), dtype=torch.float32)
    patch_embed = sam_model.image_encoder.trunk.patch_embed(input_tensor)
    img_size_pos_embed = sam_model.image_encoder.trunk._get_pos_embed(patch_embed.shape[1:3]).cpu().contiguous()
    del sam_model

    # Load and process state dict
    state_dict = torch.load(args.checkpoint, map_location="cpu")['model']
    state_dict.update(config_dict)
    state_dict['image_encoder.trunk.img_size_pos_embed'] = img_size_pos_embed

    # Determine output path
    output_path = args.output or osp.splitext(args.checkpoint)[0] + ".safetensors"
    
    # Save safetensors file
    save_file(state_dict, output_path)
    print(f"Converted {args.checkpoint} to {output_path}")

if __name__ == "__main__":
    main()