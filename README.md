# SAM2-rs: Segment Anything Model v2 (Rust Implementation)


A Rust implementation of Meta's Segment Anything Model (SAM) v2, base on [Candle](https://github.com/huggingface/candle) providing high-performance image and video segmentation capabilities with full Rust ecosystem integration.

## export weights to safetensors

```
git clone https://github.com/facebookresearch/sam2.git

# copy export_safetensor.py to sam2
cd sam2

# download pytorch weight file

python {SAM2_RS_PATH}/export_safetensor.py --config configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint path/to/sam2.1_hiera_large.pt
```

## Basic Usage

see `examples/run_image.rs` and `examples/run_video.rs` 
