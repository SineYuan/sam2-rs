use candle_core::{Module, Result, Tensor, D};
use candle_nn::{Activation, Conv2d, Conv2dConfig, Linear, VarBuilder};

use crate::modeling::position_encoding::PositionEmbeddingSine;
use crate::modeling::sam_utils::{DropPath, LayerNorm2d};

pub struct MaskDownSampler {
    encoder: Vec<Conv2d>,
    norms: Vec<LayerNorm2d>,
    activations: Vec<Activation>,
    final_conv: Conv2d,
}

impl MaskDownSampler {
    pub fn new(
        embed_dim: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        total_stride: usize,
        activation: Activation,
        vb: VarBuilder,
    ) -> Result<Self> {

        let num_layers = ((total_stride as f64).log2() / (stride as f64).log2()).floor() as usize;
        let mut encoder = Vec::new();
        let mut norms = Vec::new();
        let mut activations = Vec::new();
        let mut mask_in_chans = 1;
        let mut mask_out_chans = 1;

        let mut conv_config = candle_nn::Conv2dConfig::default();
        conv_config.stride = stride;
        conv_config.padding = padding;
        for i in 0..num_layers {
            mask_out_chans = mask_in_chans * (stride * stride);
            let conv = candle_nn::conv2d(
                mask_in_chans,
                mask_out_chans,
                kernel_size,
                conv_config,
                vb.pp(format!("encoder.{}", 3 * i)),
            )?;
            let norm = LayerNorm2d::new(
                mask_out_chans,
                1e-6,
                vb.pp(format!("encoder.{}", 3 * i + 1)),
            )?;
            encoder.push(conv);
            norms.push(norm);
            activations.push(activation.clone());
            mask_in_chans = mask_out_chans;
        }

        let final_conv = candle_nn::conv2d(
            mask_out_chans,
            embed_dim,
            1,
            Default::default(),
            vb.pp(format!("encoder.{}", 3 * num_layers)),
        )?;

        Ok(Self {
            encoder,
            norms,
            final_conv,
            activations,
        })
    }
}

impl Module for MaskDownSampler {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for (conv, (norm, act)) in self
            .encoder
            .iter()
            .zip(self.norms.iter().zip(self.activations.iter()))
        {
            x = conv.forward(&x)?;
            x = norm.forward(&x)?;
            x = act.forward(&x)?;
        }
        self.final_conv.forward(&x)
    }
}

struct CXBlock {
    dwconv: Conv2d,
    norm: LayerNorm2d,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Option<Tensor>,
    drop_path: DropPath,
    activation: Activation,
}

#[derive(Debug)]
pub struct CXBlockParams {
    pub dim: usize,
    pub kernel_size: usize,
    pub padding: usize,
    pub drop_path: f64,
    pub layer_scale_init_value: f32,
    pub use_dwconv: bool,
}

impl CXBlockParams {
    pub fn instantiate(&self, vb: VarBuilder) -> Result<CXBlock> {
        CXBlock::new(
            self.dim,
            self.kernel_size,
            self.padding,
            self.drop_path,
            self.layer_scale_init_value,
            self.use_dwconv,
            vb,
        )
    }
}

impl CXBlock {
    fn new(
        dim: usize,
        kernel_size: usize,
        padding: usize,
        drop_path: f64,
        layer_scale_init_value: f32,
        use_dwconv: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let groups = if use_dwconv { dim } else { 1 };
        let dwconv = candle_nn::conv2d(
            dim,
            dim,
            kernel_size,
            Conv2dConfig {
                padding: padding,
                groups: groups,
                ..Default::default()
            },
            vb.pp("dwconv"),
        )?;
        let norm = LayerNorm2d::new(dim, 10e-6, vb.pp("norm"))?;
        let pwconv1 = candle_nn::linear(dim, 4 * dim, vb.pp("pwconv1"))?;
        let pwconv2 = candle_nn::linear(4 * dim, dim, vb.pp("pwconv2"))?;
        let gamma = if layer_scale_init_value > 0.0 {
            Some(vb.get((dim), "gamma")?)
        } else {
            None
        };
        let drop_path = DropPath::new(drop_path, true);

        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gamma,
            drop_path,
            activation: Activation::Gelu,
        })
    }
}

impl Module for CXBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let input = x.clone();
        let mut x = self.dwconv.forward(x)?;
        x = self.norm.forward(&x)?;

        let (n, c, h, w) = x.dims4()?;
        x = x.permute((0, 2, 3, 1))?.contiguous()?;
        x = self.pwconv1.forward(&x)?;
        x = self.activation.forward(&x)?;
        x = self.pwconv2.forward(&x)?;

        if let Some(gamma) = &self.gamma {
            x = x.broadcast_mul(gamma)?;
        }

        x = x.permute((0, 3, 1, 2))?;

        let out = self.drop_path.forward(&x)?.add(&input)?;

        Ok(out)
        //self.drop_path.forward(&x)?.add(&input)?
    }
}

pub struct Fuser {
    proj: Option<Conv2d>,
    layers: Vec<CXBlock>,
}

impl Fuser {
    pub fn new(
        layer_params: CXBlockParams,
        num_layers: usize,
        dim: Option<usize>,
        input_projection: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let proj = if input_projection {
            Some(candle_nn::conv2d(
                dim.unwrap(),
                dim.unwrap(),
                1,
                Default::default(),
                vb.pp("proj"),
            )?)
        } else {
            None
        };

        let mut layers = Vec::new();
        for i in 0..num_layers {
            let layer_vb = vb.pp(format!("layers.{}", i));
            layers.push(layer_params.instantiate(layer_vb)?);
        }

        Ok(Self { proj, layers })
    }
}

impl Module for Fuser {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {

        let mut x = match &self.proj {
            Some(proj) => {
                proj.forward(x)?
            }
            None => {
                x.clone()
            }
        };

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }
}

pub struct MemoryEncoder {
    mask_downsampler: MaskDownSampler,
    pix_feat_proj: Conv2d,
    fuser: Fuser,
    position_encoding: PositionEmbeddingSine,
    out_proj: Option<Conv2d>,

    out_dim: usize,
}

impl MemoryEncoder {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        mask_downsampler: MaskDownSampler,
        fuser: Fuser,
        position_encoding: PositionEmbeddingSine,
        vb: VarBuilder,
    ) -> Result<Self> {
        let pix_feat_proj = candle_nn::conv2d(
            in_dim,
            in_dim,
            1,
            Default::default(),
            vb.pp("pix_feat_proj"),
        )?;
        let out_proj = if out_dim != in_dim {
            Some(candle_nn::conv2d(
                in_dim,
                out_dim,
                1,
                Default::default(),
                vb.pp("out_proj"),
            )?)
        } else {
            None
        };

        Ok(Self {
            mask_downsampler,
            pix_feat_proj,
            fuser,
            position_encoding,
            out_proj,
            out_dim,
        })
    }

    pub fn output_dim(&self) -> usize {
        self.out_dim
    }

    pub fn forward(
        &self,
        pix_feat: &Tensor,
        masks: &Tensor,
        skip_mask_sigmoid: bool,
    ) -> Result<(Tensor, Tensor)> {

        let masks = if skip_mask_sigmoid {
            masks
        } else {
            &candle_nn::ops::sigmoid(masks)?
        };
        let masks = self.mask_downsampler.forward(masks)?;

        let x = self.pix_feat_proj.forward(pix_feat)?;
        let x = x.add(&masks)?;
        let x = self.fuser.forward(&x)?;
        let x = match &self.out_proj {
            Some(proj) => proj.forward(&x)?,
            None => x.clone(),
        };

        let pos = self.position_encoding.forward(&x)?;
        Ok((x, pos))
    }
}
