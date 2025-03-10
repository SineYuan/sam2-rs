use candle_core::{DType, Device, Result, Shape, Tensor, D};
use candle_nn::VarBuilder;

use std::cell::RefCell;
use std::collections::HashMap;

pub struct PositionEmbeddingSine {
    num_pos_feats: usize,
    temperature: f64,
    normalize: bool,
    scale: f64,
    cache: RefCell<HashMap<(usize, usize), Tensor>>,
}

impl PositionEmbeddingSine {
    pub fn new(
        num_pos_feats: usize,
        temperature: i32,
        normalize: bool,
        scale: Option<f64>,
        warmup_cache: bool,
        image_size: usize,
        strides: &[usize],
        device: &Device,
    ) -> Self {
        assert!(num_pos_feats % 2 == 0, "Expecting even model width");
        let num_pos_feats = num_pos_feats / 2;
        let temperature = temperature as f64;

        // Corrected scale handling
        let scale = if let Some(s) = scale {
            if !normalize {
                panic!("normalize should be true if scale is passed");
            }
            s
        } else {
            2.0 * std::f64::consts::PI
        };

        let mut cache: HashMap<(usize, usize), Tensor> = HashMap::new();
        if warmup_cache {
            for &stride in strides {
                let cache_key = (image_size / stride, image_size / stride);
                let pos = Self::_pe(
                    1,
                    device,
                    cache_key,
                    num_pos_feats,
                    temperature,
                    normalize,
                    scale,
                )
                .unwrap();
                let pos_first_sample = pos.get(0).unwrap(); // Equivalent to pos[0]
                cache.insert(cache_key, pos_first_sample); // Save the output to the cache
            }

            let mut mm: HashMap<String, Tensor> = HashMap::new();
            for ((h, w), tensor) in &cache {
                let key = format!("{}_{}", h, w); // Convert cache_key to string
                mm.insert(key, tensor.clone());
            }

            candle_core::safetensors::save(&mm, "pe_cache.safetensors").unwrap();
        }

        Self {
            num_pos_feats,
            temperature,
            normalize,
            scale,
            cache: RefCell::new(cache), // Wrap the cache in a RefCell
        }
    }

    fn _encode_xy(&self, x: &Tensor, y: &Tensor) -> Result<(Tensor, Tensor)> {
        assert_eq!(x.dims(), y.dims());
        let x_embed = (x * self.scale)?;
        let y_embed = (y * self.scale)?;

        let dim_t = ((Tensor::arange(0, self.num_pos_feats as u32, &x.device())?
            .to_dtype(DType::F32)?
            / 2.0)?
            .floor()?
            * 2.0)?;
        //let dimt_t = (self.temperature.powf(2.0 * (dim_t.div(2).floor() as f64) / self.num_pos_feats as f64))?;
        let base = Tensor::new(self.temperature, &x.device())?;
        let divisor = Tensor::new(self.num_pos_feats as f32, &x.device())?;
        let dim_t = base.broadcast_pow(&dim_t)?.broadcast_div(&divisor)?;

        let pos_x = x_embed.unsqueeze(1)?.broadcast_div(&dim_t)?;
        let pos_y = y_embed.unsqueeze(1)?.broadcast_div(&dim_t)?;

        let pos_x_sin = pos_x.narrow(1, 0, pos_x.dim(1)? / 2)?.sin()?;
        let pos_x_cos = pos_x
            .narrow(1, pos_x.dim(1)? / 2, pos_x.dim(1)? / 2)?
            .cos()?;
        let pos_y_sin = pos_y.narrow(1, 0, pos_y.dim(1)? / 2)?.sin()?;
        let pos_y_cos = pos_y
            .narrow(1, pos_y.dim(1)? / 2, pos_y.dim(1)? / 2)?
            .cos()?;

        let pos_x = Tensor::stack(&[pos_x_sin, pos_x_cos], 2)?.flatten_from(1)?;
        let pos_y = Tensor::stack(&[pos_y_sin, pos_y_cos], 2)?.flatten_from(1)?;

        Ok((pos_x, pos_y))
    }

    pub fn encode_boxes(&self, x: &Tensor, y: &Tensor, w: &Tensor, h: &Tensor) -> Result<Tensor> {
        let (pos_x, pos_y) = self._encode_xy(x, y)?;
        let pos = Tensor::cat(&[&pos_y, &pos_x, &h.unsqueeze(1)?, &w.unsqueeze(1)?], 1)?;
        Ok(pos)
    }

    pub fn encode_points(&self, x: &Tensor, y: &Tensor, labels: &Tensor) -> Result<Tensor> {
        let (bx, nx) = (x.dims()[0], x.dims()[1]);
        let (by, ny) = (y.dims()[0], y.dims()[1]);
        let (bl, nl) = (labels.dims()[0], labels.dims()[1]);

        assert_eq!(bx, by);
        assert_eq!(nx, ny);
        assert_eq!(bx, bl);
        assert_eq!(nx, nl);

        let (pos_x, pos_y) = self._encode_xy(&x.flatten_all()?, &y.flatten_all()?)?;

        // Corrected reshape logic
        let pos_x = pos_x.reshape((bx, nx, pos_x.elem_count() / (bx * nx)))?;
        let pos_y = pos_y.reshape((by, ny, pos_y.elem_count() / (by * ny)))?;

        let pos = Tensor::cat(&[&pos_y, &pos_x, &labels.unsqueeze(2)?], 2)?;
        Ok(pos)
    }

    fn _pe(
        b: usize,
        device: &Device,
        cache_key: (usize, usize),
        num_pos_feats: usize,
        temperature: f64,
        normalize: bool,
        scale: f64,
    ) -> Result<Tensor> {
        let (h, w) = cache_key;

        let y_embed = Tensor::arange(1i64, (h + 1) as i64, device)?
            .to_dtype(DType::F32)? // Ensure consistent data type
            .unsqueeze(0)? // Shape: (1, H)
            .unsqueeze(2)? // Shape: (1, H, 1)
            .broadcast_as((b, h, w))?; // Shape: (B, H, W)

        let x_embed = Tensor::arange(1i64, (w + 1) as i64, device)?
            .to_dtype(DType::F32)? // Ensure consistent data type
            .unsqueeze(0)? // Shape: (1, W)
            .unsqueeze(1)? // Shape: (1, 1, W)
            .broadcast_as((b, h, w))?; // Shape: (B, H, W)

        // Normalize y_embed
        let y_last_col = y_embed.narrow(1, h - 1, 1)?; // Shape: [B, 1, W]
        let y_embed = if normalize {
            (y_embed.broadcast_div(&(y_last_col + 1e-6)?.broadcast_as(y_embed.shape())?)? * scale)?
        } else {
            y_embed
        };

        // Normalize x_embed
        let x_last_col = x_embed.narrow(2, w - 1, 1)?; // Shape: [B, H, 1]
        let x_embed = if normalize {
            (x_embed.broadcast_div(&(x_last_col + 1e-6)?.broadcast_as(x_embed.shape())?)? * scale)?
        } else {
            x_embed
        };

        let dim_t = ((Tensor::arange(0, num_pos_feats as u32, device)?.to_dtype(DType::F32)?
            / 2.0)?
            .floor()?
            * 2.0)?;

        let base = Tensor::new(temperature as f32, device)?;
        let divisor = Tensor::new(num_pos_feats as f32, device)?;
        let dim_t = base.broadcast_pow(&dim_t.broadcast_div(&divisor)?)?;

        let pos_x = x_embed.unsqueeze(3)?.broadcast_div(&dim_t)?;
        let pos_y = y_embed.unsqueeze(3)?.broadcast_div(&dim_t)?;

        // Select even and odd indices for pos_x and pos_y
        let (pos_x_sin, pos_x_cos) = select_even_odd(&pos_x, 3)?;
        let (pos_y_sin, pos_y_cos) = select_even_odd(&pos_y, 3)?;

        // Stack and flatten
        let pos_x = Tensor::stack(&[pos_x_sin, pos_x_cos], 4)?.flatten_from(3)?;
        let pos_y = Tensor::stack(&[pos_y_sin, pos_y_cos], 4)?.flatten_from(3)?;

        let pos = Tensor::cat(&[&pos_y, &pos_x], 3)?.permute((0, 3, 1, 2))?;
        Ok(pos)
    }
}

impl candle_nn::Module for PositionEmbeddingSine {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let b = xs.dims()[0];
        let cache_key = (xs.dims()[2], xs.dims()[3]);

        // Borrow the cache mutably using RefCell
        if let Some(cached) = self.cache.borrow().get(&cache_key) {
            // Corrected repeat usage
            let target_shape =
                Shape::from_dims(&[b, cached.dims()[0], cached.dims()[1], cached.dims()[2]]);
            //return Ok(cached.unsqueeze(0)?.repeat(target_shape)?);
            return Ok(cached.unsqueeze(0)?.broadcast_as(target_shape)?);
        }

        let pos = Self::_pe(
            b,
            &xs.device(),
            cache_key,
            self.num_pos_feats,
            self.temperature,
            self.normalize,
            self.scale,
        )?;

        let pos_first_sample = pos.narrow(0, 0, 1)?;
        self.cache
            .borrow_mut()
            .insert(cache_key, pos_first_sample.clone());
        Ok(pos)
    }
}

fn select_even_odd(tensor: &Tensor, dim: usize) -> Result<(Tensor, Tensor)> {
    let size = tensor.dim(dim)? as u32;
    let even_indices: Vec<u32> = (0u32..size).step_by(2).collect();
    let odd_indices: Vec<u32> = (1u32..size).step_by(2).collect();

    // Convert indices to tensors
    let even_idx_len = even_indices.len();
    let odd_idx_len = odd_indices.len();
    let even_indices_tensor = Tensor::from_vec(even_indices, (even_idx_len,), &tensor.device())?;
    let odd_indices_tensor = Tensor::from_vec(odd_indices, (odd_idx_len,), &tensor.device())?;

    // Use index_select to extract even and odd indices
    let even = tensor.index_select(&even_indices_tensor, dim)?.sin()?;
    let odd = tensor.index_select(&odd_indices_tensor, dim)?.cos()?;

    Ok((even, odd))
}

#[derive(Debug)]
pub struct PositionEmbeddingRandom {
    positional_encoding_gaussian_matrix: Tensor,
}

impl PositionEmbeddingRandom {
    pub fn new(num_pos_feats: usize, vb: VarBuilder) -> Result<Self> {
        let positional_encoding_gaussian_matrix =
            vb.get((2, num_pos_feats), "positional_encoding_gaussian_matrix")?;
        Ok(Self {
            positional_encoding_gaussian_matrix,
        })
    }

    fn pe_encoding(&self, coords: &Tensor) -> Result<Tensor> {
        let coords = coords.affine(2., -1.)?;
        let coords = coords.broadcast_matmul(&self.positional_encoding_gaussian_matrix)?;
        let coords = (coords * (2. * std::f64::consts::PI))?;
        Tensor::cat(&[coords.sin()?, coords.cos()?], D::Minus1)
    }

    pub fn forward(&self, h: usize, w: usize) -> Result<Tensor> {
        let device = self.positional_encoding_gaussian_matrix.device();
        let x_embed = (Tensor::arange(0u32, w as u32, device)?.to_dtype(DType::F32)? + 0.5)?;
        let y_embed = (Tensor::arange(0u32, h as u32, device)?.to_dtype(DType::F32)? + 0.5)?;
        let x_embed = (x_embed / w as f64)?
            .reshape((1, ()))?
            .broadcast_as((h, w))?;
        let y_embed = (y_embed / h as f64)?
            .reshape(((), 1))?
            .broadcast_as((h, w))?;
        let coords = Tensor::stack(&[&x_embed, &y_embed], D::Minus1)?;
        self.pe_encoding(&coords)?.permute((2, 0, 1))
    }

    pub fn forward_with_coords(
        &self,
        coords_input: &Tensor,
        image_size: (usize, usize),
    ) -> Result<Tensor> {
        let coords0 = (coords_input.narrow(D::Minus1, 0, 1)? / image_size.1 as f64)?;
        let coords1 = (coords_input.narrow(D::Minus1, 1, 1)? / image_size.0 as f64)?;
        let c = coords_input.dim(D::Minus1)?;
        let coords_rest = coords_input.narrow(D::Minus1, 2, c - 2)?;
        let coords = Tensor::cat(&[&coords0, &coords1, &coords_rest], D::Minus1)?;
        self.pe_encoding(&coords)
    }
}

/// Get 1D sine positional embedding as in the original Transformer paper.
pub fn get_1d_sine_pe(pos_inds: &Tensor, dim: usize, temperature: f32) -> Result<Tensor> {
    let pe_dim = dim / 2;

    let divisor = Tensor::new(pe_dim as f32, &pos_inds.device())?;
    let base = Tensor::new(temperature, &pos_inds.device())?;
    let dim_t =
        ((Tensor::arange(0f32, pe_dim as f32, &pos_inds.device())?.to_dtype(DType::F32)? / 2.0)?
            .floor()?
            * 2.0)?;
    let dim_t = dim_t.broadcast_div(&divisor)?;
    //let dimt_t = (self.temperature.powf(2.0 * (dim_t.div(2).floor() as f64) / self.num_pos_feats as f64))?;
    let dim_t = base.broadcast_pow(&dim_t)?;

    let pos_embed = pos_inds.unsqueeze(1)?.broadcast_div(&dim_t)?;

    let sin_part = pos_embed.sin()?;
    let cos_part = pos_embed.cos()?;

    let pos_embed = Tensor::cat(&[sin_part, cos_part], 1)?;

    Ok(pos_embed)
}
