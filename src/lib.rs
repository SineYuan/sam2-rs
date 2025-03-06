use candle_core::{Device, Result, Tensor};
use image::{imageops::FilterType, DynamicImage};
use ndarray::prelude::*;
use ndarray::Axis;

pub mod modeling;
pub mod sam2_image_predictor;
pub mod sam2_video_predictor;

// 如果需要，可以在这里重新导出模块中的特定类型或函数
//pub use position_encoding::{PositionEmbeddingSine, PositionEmbeddingRandom};

pub const IMAGE_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGE_STD: [f32; 3] = [0.229, 0.224, 0.225];

fn preprocess_image(
    img: &DynamicImage,
    target_size: (u32, u32),
    means: &[f32; 3],
    stds: &[f32; 3],
    device: &Device,
) -> Result<Tensor> {
    // 调整图片大小为目标尺寸，使用双线性插值
    let resized_img = img.resize_exact(target_size.0, target_size.1, FilterType::Triangle);

    // 将图片转换为 RGB 格式
    let rgb_img = resized_img.to_rgb8();

    // 创建一个 [3, height, width] 的 NDArray 并填充数据
    let (width, height) = rgb_img.dimensions();
    let mut array = Array3::<f32>::zeros((3, height as usize, width as usize));

    for (x, y, pixel) in rgb_img.enumerate_pixels() {
        let r = pixel[0] as f32 / 255.0; // 归一化到 [0, 1]
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;

        array[[0, y as usize, x as usize]] = r;
        array[[1, y as usize, x as usize]] = g;
        array[[2, y as usize, x as usize]] = b;
    }

    // 对每个通道进行归一化
    for channel in 0..3 {
        let mut channel_view = array.index_axis_mut(Axis(0), channel);
        channel_view.mapv_inplace(|val| (val - means[channel]) / stds[channel]);
    }

    // 输出为 Vec<f32>
    let data = array.into_iter().collect();

    Tensor::from_vec(data, (1, 3, width as usize, height as usize), device)
}
