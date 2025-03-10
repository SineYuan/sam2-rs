use sam2::{
    ImageEmbedding, Prompt, SAM2Base, SAM2ImagePredictor, 
};
use image::{GrayImage, io::Reader as ImageReader};
use candle_core::Device;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize device (using CPU)
    //let device = Device::Cpu;
    let device = Device::cuda_if_available(0)?;

    // 2. Load model (assuming pre-converted to Candle format)
    let checkpoint_path = "checkpoints/sam2.1_hiera_large.safetensors";
    let model = SAM2Base::from_safetensors(checkpoint_path, &device)?;
    let predictor = SAM2ImagePredictor::new(model)?;

    // 3. Load test image
    let img_path = "data/truck.jpg";
    let image = ImageReader::open(img_path)?.decode()?;

    // 4. Generate image embeddings
    let embedding = predictor.get_image_embedding(&image)?;

    // 5. Define prompts (example uses point prompts)
    let prompts = vec![
        Prompt::Point(560, 745, 0),
        Prompt::Point(560, 660, 1),
        Prompt::Box(425, 600, 700, 875),
    ];

    // 6. Run prediction
    let (masks, ious, _) = predictor.predict(
        &embedding,
        &prompts,
        0.0,    // Mask threshold
        false     // Multi-mask output
    )?;

    println!("masks {:?}", masks);
    for (i, mask) in masks.iter().enumerate() {
        let mask_path = format!("mask_{}.png", i); 

        let shape = mask.shape().dims();
        let (height, width) = (shape[0] as u32, shape[1] as u32);
        let data = mask.to_vec2::<u8>()?; 
        let pixels: Vec<u8> = data
            .into_iter()
            .flatten()
            .map(|pixel| pixel * 255) 
            .collect();

        let img = GrayImage::from_raw(width, height, pixels)
            .unwrap();
        img.save(mask_path)?;
    }
    println!("✅ Saved segmentation result to output_mask.png");
    println!("🔍 IOU scores: {:?}", ious);

    Ok(())
}
