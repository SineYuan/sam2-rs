use sam2::{
    ImageEmbedding, Prompt, SAM2Base, SAM2VideoPredictor, ImageLoader, 
};
use image::{DynamicImage, ImageBuffer, GrayImage, io::Reader as ImageReader};
use candle_core::{Device, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize device
    let device = Device::cuda_if_available(0)?;

    // 2. Load model (assuming pre-converted to Candle format)
    let checkpoint_path = "checkpoints/sam2.1_hiera_large.safetensors";
    let model = SAM2Base::from_safetensors(checkpoint_path, &device)?;
    let mut predictor = SAM2VideoPredictor::new(model, false, false, false);

    // 3. Load frames
    let video_path = "data/bedroom";
    let frame_loader = ImageLoader::new(video_path).unwrap();
    let mut infer_state = predictor.init_state(Box::new(frame_loader), false, false)?;


    let frame_id = 0;
    let obj_id = 2;
    let clear_old_points = false;
    let prompts = vec![Prompt::Point(210, 350, 1)];

    let vmasks = predictor.add_new_points_or_box(
        &mut infer_state,
        frame_id,
        obj_id,
        &prompts,
        clear_old_points,
    )?;

    let mut debug: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
    for (i, om) in vmasks.into_iter().enumerate() {
        debug.insert(format!("vmasks.{}", om.obj_id), om.mask);
    }

    infer_state.reset();

    let obj_id = 2;
    println!("add obj {} ", obj_id);
    let prompts = vec![Prompt::Point(200, 300, 1)];
    let objmasks = predictor.add_new_points_or_box(
        &mut infer_state,
        frame_id,
        obj_id,
        &prompts,
        clear_old_points,
    )?;

    for (i, om) in objmasks.into_iter().enumerate() {
        debug.insert(format!("out1_obj.{}", om.obj_id), om.mask);
    }

    println!("add obj {} points ", obj_id);
    let clear_old_points = false;
    let prompts = vec![Prompt::Point(275, 175, 0), Prompt::Point(195, 295, 1)];
    let objmasks = predictor.add_new_points_or_box(
        &mut infer_state,
        frame_id,
        obj_id,
        &prompts,
        clear_old_points,
    )?;
    for (i, om) in objmasks.into_iter().enumerate() {
        debug.insert(format!("out2_obj.{}", om.obj_id), om.mask);
    }

    let obj_id = 3;
    println!("add obj {} point", obj_id);
    let prompts = vec![Prompt::Point(400, 150, 1)];
    let objmasks = predictor.add_new_points_or_box(
        &mut infer_state,
        frame_id,
        obj_id,
        &prompts,
        clear_old_points,
    )?;
    for (i, om) in objmasks.into_iter().enumerate() {
        debug.insert(format!("out3_obj.{}", om.obj_id), om.mask);
    }

    let max_frame_idx = Some(10);
    //let max_frame_idx = None;
    let outs = predictor.propagate_in_video(&mut infer_state, None, max_frame_idx, false)?;

    for (frame_id, obj_masks) in outs {
        for om in obj_masks.into_iter() {
            debug.insert(format!("frame.{}.obj.{}", frame_id, om.obj_id), om.mask);
        }
    }

    Ok(())
}
