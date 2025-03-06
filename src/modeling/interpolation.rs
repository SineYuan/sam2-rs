use candle_core::{DType, Device, Result, Tensor};
use ndarray::{Array4, ArrayD};

// TODO use candle tensor inplememtation
pub fn bilinear_interpolate_tensor(
    input: &Tensor,  // Tensor shape [batch, channel, height, width]
    output_h: usize,
    output_w: usize,
) -> Result<Tensor> {
    let shape = input.dims();
    if shape.len() != 4 {
        return Err(candle_core::Error::Msg(format!(
            "except interpolate_tensor dims = 4"
        )));
    }
    let (batch, channels, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);

    let input_vec = input.flatten_all()?.to_vec1::<f32>()?;

    // transform to ndarray::Array4<f32>
    let input_array =
        Array4::from_shape_vec((batch, channels, in_h, in_w), input_vec).map_err(|_| {
            candle_core::Error::Msg("Failed to convert Tensor to Array4<f32>".to_string())
        })?;

    let output_array = bilinear_interpolate(&input_array, output_h, output_w);

    // transform back Tensor
    let output_tensor = Tensor::from_slice(
        output_array.as_slice().unwrap(),
        &[batch, channels, output_h, output_w],
        &input.device(),
    )?;

    Ok(output_tensor)
}

pub fn bilinear_interpolate(
    input: &Array4<f32>, // Tensor shape [batch, channel, height, width]
    output_h: usize, 
    output_w: usize,
) -> Array4<f32> {
    let (batch, channels, in_h, in_w) = input.dim();
    let mut output = Array4::<f32>::zeros((batch, channels, output_h, output_w));
    let scale_h = in_h as f32 / output_h as f32;
    let scale_w = in_w as f32 / output_w as f32;
    output.indexed_iter_mut().for_each(|((b, c, y, x), val)| {
        // input coords（align_corners=False）
        let y_in = (y as f32 + 0.5) * scale_h - 0.5;
        let x_in = (x as f32 + 0.5) * scale_w - 0.5;
        // bound
        let y_in = y_in.clamp(0.0, (in_h - 1) as f32);
        let x_in = x_in.clamp(0.0, (in_w - 1) as f32);
        // 
        let y0 = y_in.floor() as usize;
        let x0 = x_in.floor() as usize;
        let y1 = (y0 + 1).min(in_h - 1);
        let x1 = (x0 + 1).min(in_w - 1);
        // interpolation weight
        let dy = y_in - y0 as f32;
        let dx = x_in - x0 as f32;
        // 
        let f00 = input[[b, c, y0, x0]];
        let f01 = input[[b, c, y0, x1]];
        let f10 = input[[b, c, y1, x0]];
        let f11 = input[[b, c, y1, x1]];
        // bilinear
        *val = (1.0 - dx) * (1.0 - dy) * f00
            + dx * (1.0 - dy) * f01
            + (1.0 - dx) * dy * f10
            + dx * dy * f11;
    });
    output
}

fn tensor_to_ndarray(t: &Tensor) -> Result<ArrayD<f32>> {
    let t = t.to_device(&Device::Cpu)?;

    let t = t.contiguous()?;

    let t = match t.dtype() {
        DType::F32 => t,
        DType::F64 => t.to_dtype(DType::F32)?,
        DType::U8 => t.to_dtype(DType::F32)?,
        dt => {
            return Err(candle_core::Error::Msg(format!(
                "Unsupported dtype {:?}",
                dt
            )))
        }
    };

    let data = t.flatten_all()?.to_vec1::<f32>()?;

    let shape = t.dims().iter().map(|&x| x as usize).collect::<Vec<_>>();
    let arr = ArrayD::from_shape_vec(shape, data).unwrap();

    Ok(arr)
}
