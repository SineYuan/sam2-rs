use candle_core::{DType, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Activation, Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder};

use crate::modeling::sam_utils::{scaled_dot_product_attention, DropPath, MLP};

fn window_partition(x: &Tensor, window_size: usize) -> Result<(Tensor, (usize, usize))> {
    let (b, h, w, c) = x.dims4()?;

    let pad_h = (window_size - h % window_size) % window_size;
    let pad_w = (window_size - w % window_size) % window_size;

    let x = if pad_h > 0 || pad_w > 0 {
        x.pad_with_zeros(1, 0, pad_h)?.pad_with_zeros(2, 0, pad_w)?
    } else {
        x.clone()
    };

    let hp = h + pad_h;
    let wp = w + pad_w;

    let num_windows = (hp / window_size) * (wp / window_size);
    let x = x.reshape((
        b,
        hp / window_size,
        window_size,
        wp / window_size,
        window_size,
        c,
    ))?;
    let windows =
        x.permute((0, 1, 3, 2, 4, 5))?
            .reshape((b * num_windows, window_size, window_size, c))?;

    Ok((windows, (hp, wp)))
}

fn window_unpartition(
    windows: &Tensor,
    window_size: usize,
    pad_hw: (usize, usize),
    hw: (usize, usize),
) -> Result<Tensor> {
    let (hp, wp) = pad_hw;
    let (h, w) = hw;

    if hp < window_size || wp < window_size {
        return Err(candle_core::Error::Msg(format!(
            "pad_hw dimensions must be >= window_size"
        )));
    }

    let num_windows = (hp / window_size) * (wp / window_size);
    let b = windows.dim(0)? / num_windows;
    let c = windows.dim(3)?;

    // rearrange to original size
    let x = windows.reshape((
        b,
        hp / window_size,
        wp / window_size,
        window_size,
        window_size,
        c,
    ))?;
    let x = x.permute((0, 1, 3, 2, 4, 5))?.reshape((b, hp, wp, c))?;

    // remove padding
    if hp > h || wp > w {
        let x = x.narrow(1, 0, h)?.narrow(2, 0, w)?;
        Ok(x)
    } else {
        Ok(x)
    }
}

struct PatchEmbed {
    proj: Conv2d,
}

impl PatchEmbed {
    fn new(
        vb: VarBuilder,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        in_chans: usize,
        embed_dim: usize,
    ) -> Result<Self> {
        let config = Conv2dConfig {
            stride,
            padding,
            ..Default::default()
        };

        let proj = candle_nn::conv2d(in_chans, embed_dim, kernel_size, config, vb.pp("proj"))?;
        Ok(Self { proj })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.proj.forward(x)?;
        x.permute((0, 2, 3, 1))
    }
}

fn do_pool(
    x: &Tensor,
    pool_size: Option<(usize, usize)>,
    norm: Option<&LayerNorm>,
) -> Result<Tensor> {
    if let Some((pool_h, pool_w)) = pool_size {
        // (B, H, W, C) -> (B, C, H, W)
        let x = x.permute((0, 3, 1, 2))?;
        let x = x.max_pool2d((pool_h, pool_w))?;
        // (B, C, H', W') -> (B, H', W', C)
        let x = x.permute((0, 2, 3, 1))?;
        if let Some(norm_layer) = norm {
            return norm_layer.forward(&x);
        }
        return Ok(x);
    }
    Ok(x.clone())
}

// MultiScaleAttention 模块
struct MultiScaleAttention {
    dim: usize,
    dim_out: usize,
    num_heads: usize,
    pool_size: Option<(usize, usize)>,
    qkv: Linear,
    proj: Linear,
}

impl MultiScaleAttention {
    fn new(
        vb: VarBuilder,
        dim: usize,
        dim_out: usize,
        num_heads: usize,
        pool_size: Option<(usize, usize)>,
    ) -> Result<Self> {
        // qkv attention (dim -> dim_out * 3)
        let qkv = candle_nn::linear(dim, dim_out * 3, vb.pp("qkv"))?;

        // proj layer (dim_out -> dim_out)
        let proj = candle_nn::linear(dim_out, dim_out, vb.pp("proj"))?;

        Ok(Self {
            dim,
            dim_out,
            num_heads,
            pool_size,
            qkv,
            proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, h, w, _) = x.dims4()?;
        let total_tokens = h * w;
        let head_dim = self.dim_out / self.num_heads;

        let qkv = self
            .qkv
            .forward(x)?
            .reshape((b, total_tokens, 3, self.num_heads, head_dim))?;

        let q = qkv.narrow(2, 0, 1)?.squeeze(2)?; // 提取第 0 个维度
        let k = qkv.narrow(2, 1, 1)?.squeeze(2)?; // 提取第 1 个维度
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?; // 提取第 2 个维度

        let (mut h_new, mut w_new) = (h, w);

        let q = if let Some((pool_h, pool_w)) = self.pool_size {
            let mut q = q.reshape((b, h, w, self.num_heads * head_dim))?;
            q = do_pool(&q, Some((pool_h, pool_w)), None)?;
            let dims = q.dims4()?;
            h_new = dims.1;
            w_new = dims.2;
            q = q.reshape((b, h_new * w_new, self.num_heads, head_dim))?;
            q
        } else {
            q
        };

        let x = scaled_dot_product_attention(
            &q.transpose(1, 2)?.contiguous()?,
            &k.transpose(1, 2)?.contiguous()?,
            &v.transpose(1, 2)?.contiguous()?,
        )?;

        let x = x.transpose(1, 2)?;
        let x = x.reshape((b, h_new, w_new, self.num_heads * head_dim))?;
        self.proj.forward(&x)
    }
}

fn _scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (dim as f64).sqrt();
    let mut batch_dims = q.dims().to_vec();
    batch_dims.pop();
    batch_dims.pop();
    let q = q.flatten_to(batch_dims.len() - 1)?;
    let k = k.flatten_to(batch_dims.len() - 1)?;
    let v = v.flatten_to(batch_dims.len() - 1)?;
    let attn_weights = (q.matmul(&k.t()?)? * scale_factor)?;
    let attn_scores = candle_nn::ops::softmax_last_dim(&attn_weights)?.matmul(&v)?;
    batch_dims.push(attn_scores.dim(D::Minus2)?);
    batch_dims.push(attn_scores.dim(D::Minus1)?);
    attn_scores.reshape(batch_dims)
}

struct MultiScaleBlock {
    dim: usize,
    dim_out: usize,
    norm1: LayerNorm,
    window_size: usize,
    pool_size: Option<(usize, usize)>,
    attn: MultiScaleAttention,
    drop_path: DropPath,
    norm2: LayerNorm,
    mlp: MLP,
    proj: Option<Linear>,
}

impl MultiScaleBlock {
    pub fn new(
        vb: VarBuilder,
        dim: usize,
        dim_out: usize,
        num_heads: usize,
        mlp_ratio: f64,
        drop_path: f64,
        q_stride: Option<(usize, usize)>,
        act_layer: Activation,
        window_size: usize,
    ) -> Result<Self> {
        let pool_size = q_stride;

        let mut layer_norm_conf = candle_nn::LayerNormConfig::default();
        layer_norm_conf.eps = 10e-6;
        let norm1 = candle_nn::layer_norm(dim, 1e-6, vb.pp("norm1"))?;
        let norm2 = candle_nn::layer_norm(dim_out, 1e-6, vb.pp("norm2"))?;

        // 初始化注意力模块
        let attn_vb = vb.pp("attn");
        let attn = MultiScaleAttention::new(attn_vb, dim, dim_out, num_heads, pool_size)?;

        // DropPath 初始化
        let drop_path = if drop_path > 0.0 {
            DropPath::new(drop_path, true)
        } else {
            DropPath::identity()
        };

        // MLP 初始化
        let mlp_vb = vb.pp("mlp");
        let mlp = MLP::new(
            mlp_vb,
            dim_out,
            (dim_out as f64 * mlp_ratio) as usize,
            dim_out,
            2,
            act_layer,
            false,
        )?;

        // 如果维度不同，则初始化投影层
        let proj = if dim != dim_out {
            let proj: Linear = candle_nn::linear(dim, dim_out, vb.pp("proj"))?;
            Some(proj)
        } else {
            None
        };

        Ok(Self {
            dim,
            dim_out,
            norm1: norm1,
            window_size,
            pool_size,
            attn,
            drop_path,
            norm2: norm2,
            mlp,
            proj,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shortcut = x.clone();
        let x = self.norm1.forward(x)?;

        // Skip connection
        let shortcut = if let Some(proj) = &self.proj {
            do_pool(&proj.forward(&x)?, self.pool_size, None)?
        } else {
            shortcut
        };

        // Window partition
        let mut window_size = self.window_size;
        let dims = x.dims4()?; // 获取输入张量的形状
        let (mut h, mut w) = (dims.1, dims.2); // 提取高度和宽度
        let (mut pad_h, mut pad_w) = (0, 0);
        let mut x_partitioned = x.clone();

        if window_size > 0 {
            let (x_p, pad_hw) = window_partition(&x, window_size)?;
            x_partitioned = x_p;
            (pad_h, pad_w) = pad_hw;
        }

        // Window Attention + Q Pooling (if stage change)
        let x_attn = self.attn.forward(&x_partitioned)?;

        if self.pool_size.is_some() {
            window_size /= self.pool_size.unwrap().0;
            let dims = shortcut.dims4()?; // 获取新的维度
            h = dims.1; // 第二个维度是高度
            w = dims.2; // 第三个维度是宽度
            pad_h = (window_size - h % window_size) % window_size;
            pad_w = (window_size - w % window_size) % window_size;
            pad_h += h;
            pad_w += w;
        }

        // Reverse window partition
        let x_unpartitioned = {
            if self.window_size > 0 {
                window_unpartition(&x_attn, window_size, (pad_h, pad_w), (h, w))?
            } else {
                x_attn
            }
        };

        let x = (shortcut + self.drop_path.forward(&x_unpartitioned)?)?;
        let x_mlp = self.mlp.forward(&self.norm2.forward(&x)?)?;
        Ok((x + self.drop_path.forward(&x_mlp)?)?)
    }
}

#[derive(Debug)]
pub struct HieraConfig {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub drop_path_rate: f64,
    pub q_pool: usize,
    pub q_stride: (usize, usize),
    pub stages: Vec<usize>,
    pub dim_mul: f64,
    pub head_mul: f64,
    pub window_pos_embed_bkg_spatial_size: (usize, usize),
    pub window_spec: Vec<usize>,
    pub global_att_blocks: Vec<usize>,
    pub return_interm_layers: bool,
}

impl Default for HieraConfig {
    fn default() -> Self {
        Self {
            embed_dim: 96,                               // initial embed dim
            num_heads: 1,                                // initial number of heads
            drop_path_rate: 0.0,                         // stochastic depth
            q_pool: 3,                                   // number of q_pool stages
            q_stride: (2, 2),                            // downsample stride between stages
            stages: vec![2, 3, 16, 3],                   // blocks per stage
            dim_mul: 2.0,                                // dim_mul factor at stage shift
            head_mul: 2.0,                               // head_mul factor at stage shift
            window_pos_embed_bkg_spatial_size: (14, 14), // background spatial size
            window_spec: vec![8, 4, 14, 7],              // window size per stage
            global_att_blocks: vec![12, 16, 20],         // global attention blocks
            return_interm_layers: true,                  // return features from every stage
        }
    }
}

pub struct Hiera {
    patch_embed: PatchEmbed,
    img_size_pos_embed: Option<Tensor>,
    pos_embed: Tensor,
    pos_embed_window: Tensor,
    blocks: Vec<MultiScaleBlock>,
    stage_ends: Vec<usize>,
    q_pool_blocks: Vec<usize>,
    return_interm_layers: bool,

    pub channel_list: Vec<usize>,
}

impl Hiera {
    pub fn new(
        vb: VarBuilder,
        conf: HieraConfig,
        img_size_pos_embed: Option<Tensor>,
    ) -> Result<Self> {
        let embed_dim = conf.embed_dim;

        // 初始化 Patch Embedding
        let patch_embed = PatchEmbed::new(
            vb.pp("patch_embed"),
            7, // kernel_size
            4, // stride
            3, // padding
            3, // 输入通道数（RGB 图像）
            embed_dim,
        )?;

        // 初始化位置嵌入参数
        let pos_embed = vb.get(
            (
                1,
                embed_dim,
                conf.window_pos_embed_bkg_spatial_size.0,
                conf.window_pos_embed_bkg_spatial_size.1,
            ),
            "pos_embed",
        )?;
        let pos_embed_window = vb.get(
            (1, embed_dim, conf.window_spec[0], conf.window_spec[0]),
            "pos_embed_window",
        )?;

        let depth = conf.stages.iter().sum();
        let dpr = (0..depth)
            .map(|x| conf.drop_path_rate * (x as f64) / ((depth - 1) as f64))
            .collect::<Vec<_>>();

        // stage_ends
        let mut stage_ends = Vec::new();
        let mut cumulative_blocks = 0;
        for &stage in &conf.stages {
            cumulative_blocks += stage;
            stage_ends.push(cumulative_blocks - 1);
        }

        // q_pool_blocks
        let q_pool_blocks: Vec<usize> = if conf.q_pool > 0 {
            stage_ends[..conf.q_pool.min(stage_ends.len())]
                .iter()
                .map(|&x| x + 1)
                .collect()
        } else {
            Vec::new()
        };

        // Multi-Scale Blocks
        let mut blocks = Vec::new();
        let mut dim = embed_dim;
        let mut cur_stage = 1;
        let mut num_heads = conf.num_heads;

        for i in 0..depth {
            let window_size = if conf.global_att_blocks.contains(&i) {
                0 // global attention
            } else {
                conf.window_spec[cur_stage - 1]
            };

            let mut dim_out = dim;
            if stage_ends.contains(&(i - 1)) {
                //let dim_out = if i == stage_ends[cur_stage - 1] {
                dim_out = (dim as f64 * conf.dim_mul) as usize;
                num_heads = (num_heads * conf.head_mul as usize) as usize;
                cur_stage += 1;
            };

            let block_vb = vb.pp(&format!("blocks.{}", i));
            let block = MultiScaleBlock::new(
                block_vb,
                dim,
                dim_out,
                num_heads,
                4.0, // mlp_ratio
                dpr[i],
                if q_pool_blocks.contains(&i) {
                    Some(conf.q_stride)
                } else {
                    None
                },
                Activation::Gelu,
                window_size,
            )?;
            blocks.push(block);

            dim = dim_out;
        }

        // 计算 channel_list
        let channel_list = if conf.return_interm_layers {
            stage_ends
                .iter()
                .rev()
                .map(|&i| blocks[i].dim_out)
                .collect::<Vec<_>>()
        } else {
            vec![blocks.last().unwrap().dim_out]
        };

        Ok(Self {
            patch_embed,
            pos_embed,
            pos_embed_window,
            img_size_pos_embed,
            blocks,
            stage_ends,
            q_pool_blocks,
            return_interm_layers: conf.return_interm_layers,
            channel_list,
        })
    }

    fn _get_pos_embed(&self, hw: (usize, usize)) -> Result<Tensor> {
        let (h, w) = hw;

        let pos_embed = bicubic_interpolation(&self.pos_embed, (h, w))?;


        let pos_embed_shape: Vec<usize> = pos_embed.dims().to_vec();
        let window_embed_shape: Vec<usize> = self.pos_embed_window.dims().to_vec();

        let repeats: Vec<usize> = pos_embed_shape
            .iter()
            .zip(window_embed_shape.iter())
            .map(|(x, y)| x / y)
            .collect();

        // tile
        let tiled_window_embed = tile(&self.pos_embed_window, &repeats)?;

        let pos_embed = (pos_embed + tiled_window_embed)?;

        // reshape to (B, H, W, C)
        pos_embed.permute((0, 2, 3, 1))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        // Patch Embedding
        let mut x = self.patch_embed.forward(x)?;

        // add pos embedding
        x = if let Some(img_size_pos_embed) = self.img_size_pos_embed.as_ref() {
            (x + img_size_pos_embed)?
        } else {
            let hw = (x.dim(1)?, x.dim(2)?);
            let pos_embed = self._get_pos_embed(hw)?;
            (x + &pos_embed)?
        };

        //  Multi-Scale Blocks
        let mut outputs = Vec::new();
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x)?;

            if (&i == self.stage_ends.last().unwrap())
                || (self.return_interm_layers && self.stage_ends.contains(&i))
            {
                let feats = x.permute((0, 3, 1, 2))?; // 转换为 (B, C, H, W)
                outputs.push(feats);
            }
        }

        Ok(outputs)
    }
}

fn tile(tensor: &Tensor, repeats: &[usize]) -> Result<Tensor> {
    let original_shape: Vec<usize> = tensor.dims().to_vec();

    if original_shape.len() != repeats.len() {
        return Err(candle_core::Error::Msg(
            "The length of repeats must match the number of dimensions in the tensor.".to_string(),
        ));
    }

    tensor.repeat(repeats)
}

// can not alias with torch.nn.functional.interpolate
fn bicubic_interpolation(input: &Tensor, size: (usize, usize)) -> Result<Tensor> {
    let (h, w) = size;
    let (in_h, in_w) = {
        let shape = input.shape().dims();
        (shape[2], shape[3]) // [B, C, H, W]
    };

    let height_scale = in_h as f64 / h as f64;
    let width_scale = in_w as f64 / w as f64;

    let mut output_data = Vec::new();

    for y_out in 0..h {
        for x_out in 0..w {
            let y_in = (y_out as f64 + 0.5) * height_scale - 0.5;
            let x_in = (x_out as f64 + 0.5) * width_scale - 0.5;

            let y0 = y_in.floor() as isize;
            let x0 = x_in.floor() as isize;

            let weights = compute_bicubic_weights(y_in - y0 as f64, x_in - x0 as f64);

            let mut interpolated_value = Tensor::zeros(
                (input.dim(0)?, input.dim(1)?),
                input.dtype(),
                input.device(),
            )?;
            for i in -1..=2 {
                for j in -1..=2 {
                    let y = (y0 + i).clamp(0, in_h as isize - 1) as usize;
                    let x = (x0 + j).clamp(0, in_w as isize - 1) as usize;
                    let weight = weights[(i + 1) as usize][(j + 1) as usize];

                    let weight_tensor = match input.dtype() {
                        DType::F32 => Tensor::from_slice(&[weight as f32], &[], input.device())?,
                        DType::F64 => Tensor::from_slice(&[weight], &[], input.device())?,
                        _ => panic!("Unsupported dtype"),
                    };

                    let pixel = input.i((.., .., y, x))?;
                    let weighted_pixel = pixel.broadcast_mul(&weight_tensor)?;
                    interpolated_value = interpolated_value.add(&weighted_pixel)?;
                }
            }

            output_data.push(interpolated_value);
        }
    }

    let output = Tensor::stack(&output_data, 2)?;
    let output = Tensor::reshape(&output, (input.dim(0)?, input.dim(1)?, h, w))?;

    Ok(output)
}

fn compute_bicubic_weights(dy: f64, dx: f64) -> [[f64; 4]; 4] {
    let cubic = |t: f64| {
        if t.abs() <= 1.0 {
            (1.5 * t.powi(3) - 2.5 * t.powi(2) + 1.0)
        } else if t.abs() < 2.0 {
            (-0.5 * t.powi(3) + 2.5 * t.powi(2) - 4.0 * t.abs() + 2.0)
        } else {
            0.0
        }
    };

    let mut weights = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            weights[i][j] = cubic((i as f64 - 1.0 - dy).abs()) * cubic((j as f64 - 1.0 - dx).abs());
        }
    }
    weights
}
