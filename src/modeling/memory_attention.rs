use candle_core::{Module, Result, Tensor};
use candle_nn::{Dropout, LayerNorm, Linear, VarBuilder};

use crate::modeling::sam::transformer::RoPEAttention;

#[derive(Clone, Debug)]
pub enum Activation {
    Relu,
    Gelu,
    //Glu,
}

impl Activation {
    fn apply(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Activation::Relu => xs.relu(),
            Activation::Gelu => xs.gelu(),
            //Activation::Glu => xs.glu(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RoPEAttentionParams {
    pub embedding_dim: usize,
    pub num_heads: usize,
    pub downsample_rate: usize,
    pub rope_theta: f64,
    pub rope_k_repeat: bool,
    pub feat_sizes: (usize, usize),
    pub kv_in_dim: Option<usize>,
}

impl RoPEAttentionParams {
    pub fn instantiate(&self, vb: VarBuilder) -> Result<RoPEAttention> {
        RoPEAttention::new(
            self.embedding_dim,
            self.num_heads,
            self.downsample_rate,
            self.rope_theta as f32,
            self.rope_k_repeat,
            self.feat_sizes,
            self.kv_in_dim,
            vb,
        )
    }
}

#[derive(Debug)]
pub struct MemoryAttentionLayerParams {
    pub d_model: usize,
    pub dim_feedforward: usize,
    pub dropout: f32,
    pub activation: Activation,
    pub self_attn: RoPEAttentionParams,
    pub cross_attn_image: RoPEAttentionParams,
    pub pos_enc_at_attn: bool,
    pub pos_enc_at_cross_attn_queries: bool,
    pub pos_enc_at_cross_attn_keys: bool,
}

impl MemoryAttentionLayerParams {
    pub fn instantiate(&self, vb: VarBuilder) -> Result<MemoryAttentionLayer> {
        let self_attn = self.self_attn.instantiate(vb.pp("self_attn"))?;
        let cross_attn_image = self
            .cross_attn_image
            .instantiate(vb.pp("cross_attn_image"))?;

        MemoryAttentionLayer::new(
            self.d_model,
            self.dim_feedforward,
            self.dropout,
            self.activation.clone(),
            self_attn,
            cross_attn_image,
            self.pos_enc_at_attn,
            self.pos_enc_at_cross_attn_queries,
            self.pos_enc_at_cross_attn_keys,
            vb,
        )
    }
}

#[derive(Clone)]
struct MemoryAttentionLayer {
    self_attn: RoPEAttention,
    cross_attn_image: RoPEAttention,
    linear1: Linear,
    linear2: Linear,
    dropout: Dropout,
    dropout1: Dropout,
    dropout2: Dropout,
    dropout3: Dropout,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    activation: Activation,
    pos_enc_at_attn: bool,
    pos_enc_at_cross_attn_queries: bool,
    pos_enc_at_cross_attn_keys: bool,
}

impl MemoryAttentionLayer {
    fn new(
        d_model: usize,
        dim_feedforward: usize,
        dropout: f32,
        activation: Activation,
        self_attention: RoPEAttention,
        cross_attn_image: RoPEAttention,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_queries: bool,
        pos_enc_at_cross_attn_keys: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let linear1 = candle_nn::linear(d_model, dim_feedforward, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear(dim_feedforward, d_model, vb.pp("linear2"))?;
        let dropout_module = Dropout::new(dropout);
        let norm1 = candle_nn::layer_norm(
            d_model,
            candle_nn::LayerNormConfig::default(),
            vb.pp("norm1"),
        )?;
        let norm2 = candle_nn::layer_norm(
            d_model,
            candle_nn::LayerNormConfig::default(),
            vb.pp("norm2"),
        )?;
        let norm3 = candle_nn::layer_norm(
            d_model,
            candle_nn::LayerNormConfig::default(),
            vb.pp("norm3"),
        )?;

        Ok(Self {
            self_attn: self_attention,
            cross_attn_image: cross_attn_image,
            linear1,
            linear2,
            dropout: dropout_module,
            dropout1: Dropout::new(dropout),
            dropout2: Dropout::new(dropout),
            dropout3: Dropout::new(dropout),
            norm1,
            norm2,
            norm3,
            activation,
            pos_enc_at_attn,
            pos_enc_at_cross_attn_queries,
            pos_enc_at_cross_attn_keys,
        })
    }

    fn _forward_sa(&self, tgt: &Tensor, query_pos: Option<&Tensor>) -> Result<Tensor> {
        let tgt2 = self.norm1.forward(tgt)?;
        let qk = if self.pos_enc_at_attn {
            tgt2.add(query_pos.unwrap())?
        } else {
            tgt2.clone()
        };

        let tgt2 = self.self_attn.forward(&qk, &qk, &tgt2, 0)?;
        let ret = tgt.add(&self.dropout1.forward(&tgt2, false)?)?;

        Ok(ret)
        //tgt.add(&self.dropout1.forward(&tgt2, false)?)
    }

    fn _forward_ca(
        &self,
        tgt: &Tensor,
        memory: &Tensor,
        query_pos: Option<&Tensor>,
        pos: Option<&Tensor>,
        num_k_exclude_rope: usize,
    ) -> Result<Tensor> {
        let tgt2 = self.norm2.forward(tgt)?;
        let q = if self.pos_enc_at_cross_attn_queries {
            tgt2.add(query_pos.unwrap())?
        } else {
            tgt2.clone()
        };
        let k = if self.pos_enc_at_cross_attn_keys {
            memory.add(pos.unwrap())?
        } else {
            memory.clone()
        };
        let tgt2 = self
            .cross_attn_image
            .forward(&q, &k, memory, num_k_exclude_rope)?;
        let res = tgt.add(&self.dropout2.forward(&tgt2, false)?)?;
        Ok(res)
    }

    fn forward(
        &self,
        tgt: &Tensor,
        memory: &Tensor,
        pos: Option<&Tensor>,
        query_pos: Option<&Tensor>,
        num_k_exclude_rope: usize,
    ) -> Result<Tensor> {
        let tgt = self._forward_sa(tgt, query_pos)?;
        let tgt = self._forward_ca(&tgt, memory, query_pos, pos, num_k_exclude_rope)?;

        let tgt2 = self.norm3.forward(&tgt)?;
        let tgt2 = self.linear1.forward(&tgt2)?;
        let tgt2 = self.activation.apply(&tgt2)?;
        let tgt2 = self.dropout.forward(&tgt2, false)?;
        let tgt2 = self.linear2.forward(&tgt2)?;

        let out = tgt.add(&self.dropout3.forward(&tgt2, false)?)?;

        Ok(out)
    }
}

pub struct MemoryAttention {
    layers: Vec<MemoryAttentionLayer>,
    norm: LayerNorm,
    pos_enc_at_input: bool,
    batch_first: bool,
}

impl MemoryAttention {
    pub fn new(
        d_model: usize,
        pos_enc_at_input: bool,
        layer_params: MemoryAttentionLayerParams,
        num_layers: usize,
        batch_first: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer_vb = vb.pp(format!("layers.{}", i));
            let layer = layer_params.instantiate(layer_vb)?;
            layers.push(layer);
        }
        let norm = candle_nn::layer_norm(
            d_model,
            candle_nn::LayerNormConfig::default(),
            vb.pp("norm"),
        )?;
        Ok(Self {
            layers,
            norm,
            pos_enc_at_input,
            batch_first,
        })
    }

    pub fn forward(
        &self,
        curr: &Tensor,
        curr_pos: Option<&Tensor>,
        memory: &Tensor,
        memory_pos: Option<&Tensor>,
        num_obj_ptr_tokens: usize,
    ) -> Result<Tensor> {

        let mut output = curr.clone();
        if self.pos_enc_at_input {
            if let Some(cp) = curr_pos {
                output = output.add(&(cp * 0.1)?)?;
            }
        }

        let (mut output, curr_pos, memory, memory_pos) = if self.batch_first {
            (
                output.transpose(0, 1)?,
                curr_pos.map(|t| t.transpose(0, 1)).transpose()?,
                memory.transpose(0, 1)?,
                memory_pos.map(|t| t.transpose(0, 1)).transpose()?,
            )
        } else {
            (
                output,
                curr_pos.cloned(),
                memory.clone(),
                memory_pos.cloned(),
            )
        };

        for (i, layer) in self.layers.iter().enumerate() {
            output = layer.forward(
                &output,
                &memory,
                memory_pos.as_ref(),
                curr_pos.as_ref(),
                num_obj_ptr_tokens,
            )?;
        }

        let mut output = self.norm.forward(&output)?;
        if self.batch_first {
            output = output.transpose(0, 1)?;
        }

        Ok(output)
    }
}
