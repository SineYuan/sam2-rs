use candle_core::{Result, Tensor, D};
use candle_nn::{Activation, Linear, Module, VarBuilder};

pub struct DropPath {
    drop_prob: f64,
    scale_by_keep: bool,
}

impl DropPath {
    pub fn new(drop_prob: f64, scale_by_keep: bool) -> Self {
        Self {
            drop_prob,
            scale_by_keep,
        }
    }

    pub fn identity() -> Self {
        Self {
            drop_prob: 0.0,       
            scale_by_keep: false,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        assert!(self.drop_prob == 0.0);
        return Ok(x.clone());

        // should not be here
        /*
        if self.drop_prob == 0.0 {
            return Ok(x.clone());
        }

        let keep_prob = 1.0 - self.drop_prob;
        let shape = vec![x.dim(0)?]
            .into_iter()
            .chain(std::iter::repeat(1).take(x.ndim()? - 1))
            .collect::<Vec<usize>>();

        // (bernoulli distribution)
        let random_tensor = Tensor::rand(0.0..1.0, shape, x.device())?
            .ge(keep_prob)?
            .to_dtype(DType::F32)?;

        let random_tensor = if keep_prob > 0.0 && self.scale_by_keep {
            random_tensor / keep_prob
        } else {
            random_tensor
        };

        Ok(x * &random_tensor?)
        */
    }
}
pub struct MLP {
    layers: Vec<Linear>,
    num_layers: usize,
    act: Activation,
    sigmoid_output: bool,
}

impl MLP {
    pub fn new(
        vb: VarBuilder,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        num_layers: usize,
        activation: Activation,
        sigmoid_output: bool,
    ) -> Result<Self> {
        let mut layers = Vec::new();

        let h = vec![hidden_dim; num_layers - 1];
        let dims = std::iter::once(input_dim)
            .chain(h.clone().into_iter())
            .zip(h.into_iter().chain(std::iter::once(output_dim)));

        for (i, (in_dim, out_dim)) in dims.enumerate() {
            let layer = candle_nn::linear(in_dim, out_dim, vb.pp(&format!("layers.{}", i)))?;
            layers.push(layer);
        }

        Ok(Self {
            layers,
            num_layers,
            act: activation,
            sigmoid_output,
        })
    }

    pub fn _forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            if i < self.num_layers - 1 {
                x = self.act.forward(&x)?;
            }
        }

        if self.sigmoid_output {
            x = Activation::Sigmoid.forward(&x)?;
        }

        Ok(x)
    }
}

impl Module for MLP {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self._forward(input)
    }
}

#[derive(Debug)]
pub struct LayerNorm2d {
    weight: Tensor,
    bias: Tensor,
    num_channels: usize,
    eps: f64,
}

impl LayerNorm2d {
    pub fn new(num_channels: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(num_channels, "weight")?;
        let bias = vb.get(num_channels, "bias")?;
        Ok(Self {
            weight,
            bias,
            num_channels,
            eps,
        })
    }
}

impl Module for LayerNorm2d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let u = xs.mean_keepdim(1)?; // dim=1 mean
        let xs_centered = xs.broadcast_sub(&u)?; // x - u

        let s = xs_centered.sqr()?.mean_keepdim(1)?; // dim=1 mean

        // norm
        let denominator = (s + self.eps)?.sqrt()?;
        let xs_norm = xs_centered.broadcast_div(&denominator)?;

        //（Python [:, None, None] broadcast）
        let weight_4d = self.weight.reshape((1, self.num_channels, 1, 1))?;
        let bias_4d = self.bias.reshape((1, self.num_channels, 1, 1))?;

        xs_norm.broadcast_mul(&weight_4d)?.broadcast_add(&bias_4d)
    }
}

pub struct Identity {}

impl Identity {
    pub fn new() -> Self {
        Identity {}
    }
}

impl Module for Identity {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.clone())
    }
}

pub fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (dim as f64).sqrt();
    let attn_weights = (q.matmul(&k.t()?)? * scale_factor)?;
    candle_nn::ops::softmax_last_dim(&attn_weights)?.matmul(v)
}
