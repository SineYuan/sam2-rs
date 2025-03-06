use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{layer_norm, LayerNorm, Linear, Module, VarBuilder};

use crate::modeling::sam_utils::{scaled_dot_product_attention, MLP};

#[derive(Debug, Clone)]
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    internal_dim: usize,
}

impl Attention {
    pub fn new(
        embedding_dim: usize,
        num_heads: usize,
        downsample_rate: usize,
        kv_in_dim: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let internal_dim = embedding_dim / downsample_rate;
        let kv_in_dim = kv_in_dim.unwrap_or(embedding_dim);
        let q_proj = candle_nn::linear(embedding_dim, internal_dim, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(kv_in_dim, internal_dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(kv_in_dim, internal_dim, vb.pp("v_proj"))?;
        let out_proj = candle_nn::linear(internal_dim, embedding_dim, vb.pp("out_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            internal_dim,
        })
    }

    pub fn internal_dim(&self) -> usize {
        self.internal_dim
    }

    fn separate_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, c) = x.dims3()?;
        x.reshape((b, n, self.num_heads, c / self.num_heads))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn recombine_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n_heads, n_tokens, c_per_head) = x.dims4()?;
        x.transpose(1, 2)?
            .reshape((b, n_tokens, n_heads * c_per_head))
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let q = self.q_proj.forward(q)?;
        let k = self.k_proj.forward(k)?;
        let v = self.v_proj.forward(v)?;

        let q = self.separate_heads(&q)?;
        let k = self.separate_heads(&k)?;
        let v = self.separate_heads(&v)?;

        let scale = (self.internal_dim as f64 / self.num_heads as f64).sqrt();
        let attn = q.matmul(&k.transpose(2, 3)?)?;
        let attn = (attn / scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        let out = attn.matmul(&v)?;
        let out = self.recombine_heads(&out)?;
        self.out_proj.forward(&out)
    }
}

//#[derive(Debug)]
struct TwoWayAttentionBlock {
    self_attn: Attention,
    norm1: LayerNorm,
    cross_attn_token_to_image: Attention,
    norm2: LayerNorm,
    mlp: MLP,
    norm3: LayerNorm,
    norm4: LayerNorm,
    cross_attn_image_to_token: Attention,
    skip_first_layer_pe: bool,
}

impl TwoWayAttentionBlock {
    fn new(
        embedding_dim: usize,
        num_heads: usize,
        mlp_dim: usize,
        skip_first_layer_pe: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1 = layer_norm(embedding_dim, 1e-5, vb.pp("norm1"))?;
        let norm2 = layer_norm(embedding_dim, 1e-5, vb.pp("norm2"))?;
        let norm3 = layer_norm(embedding_dim, 1e-5, vb.pp("norm3"))?;
        let norm4 = layer_norm(embedding_dim, 1e-5, vb.pp("norm4"))?;
        let self_attn = Attention::new(embedding_dim, num_heads, 1, None, vb.pp("self_attn"))?;
        let cross_attn_token_to_image = Attention::new(
            embedding_dim,
            num_heads,
            2,
            None,
            vb.pp("cross_attn_token_to_image"),
        )?;
        let cross_attn_image_to_token = Attention::new(
            embedding_dim,
            num_heads,
            2,
            None,
            vb.pp("cross_attn_image_to_token"),
        )?;
        let mlp = MLP::new(
            vb.pp("mlp"),
            embedding_dim,
            mlp_dim,
            embedding_dim,
            2,
            candle_nn::Activation::Relu,
            false,
        )?;
        Ok(Self {
            self_attn,
            norm1,
            cross_attn_image_to_token,
            norm2,
            mlp,
            norm3,
            norm4,
            cross_attn_token_to_image,
            skip_first_layer_pe,
        })
    }

    fn forward(
        &self,
        queries: &Tensor,
        keys: &Tensor,
        query_pe: &Tensor,
        key_pe: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Self attention block
        let queries = if self.skip_first_layer_pe {
            self.self_attn.forward(queries, queries, queries)?
        } else {
            let q = (queries + query_pe)?;
            let attn_out = self.self_attn.forward(&q, &q, queries)?;
            (queries + attn_out)?
        };
        let queries = self.norm1.forward(&queries)?;

        // Cross attention block, tokens attending to image embedding
        let q = (&queries + query_pe)?;
        let k = (keys + key_pe)?;
        let attn_out = self.cross_attn_token_to_image.forward(&q, &k, keys)?;
        let queries = (&queries + attn_out)?;
        let queries = self.norm2.forward(&queries)?;

        // MLP block
        let mlp_out = self.mlp.forward(&queries);
        let queries = (queries + mlp_out)?;
        let queries = self.norm3.forward(&queries)?;

        // Cross attention block, image embedding attending to tokens
        let q = (&queries + query_pe)?;
        let k = (keys + key_pe)?;
        let attn_out = self.cross_attn_image_to_token.forward(&k, &q, &queries)?;
        let keys = (keys + attn_out)?;
        let keys = self.norm4.forward(&keys)?;

        Ok((queries, keys))
    }
}

pub struct TwoWayTransformer {
    layers: Vec<TwoWayAttentionBlock>,
    final_attn_token_to_image: Attention,
    norm_final_attn: LayerNorm,
}

impl TwoWayTransformer {
    pub fn new(
        depth: usize,
        embedding_dim: usize,
        num_heads: usize,
        mlp_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb_l = vb.pp("layers");
        let mut layers = Vec::with_capacity(depth);
        for i in 0..depth {
            let layer =
                TwoWayAttentionBlock::new(embedding_dim, num_heads, mlp_dim, i == 0, vb_l.pp(i))?;
            layers.push(layer)
        }
        let final_attn_token_to_image = Attention::new(
            embedding_dim,
            num_heads,
            2,
            None,
            vb.pp("final_attn_token_to_image"),
        )?;
        let norm_final_attn = layer_norm(embedding_dim, 1e-5, vb.pp("norm_final_attn"))?;
        Ok(Self {
            layers,
            final_attn_token_to_image,
            norm_final_attn,
        })
    }

    pub fn forward(
        &self,
        image_embedding: &Tensor,
        image_pe: &Tensor,
        point_embedding: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let image_embedding = image_embedding.flatten_from(2)?.permute((0, 2, 1))?;
        let image_pe = image_pe.flatten_from(2)?.permute((0, 2, 1))?;

        let mut queries = point_embedding.clone();
        let mut keys = image_embedding;

        for layer in self.layers.iter() {
            (queries, keys) = layer.forward(&queries, &keys, point_embedding, &image_pe)?
        }

        let q = (&queries + point_embedding)?;
        let k = (&keys + image_pe)?;
        let attn_out = self.final_attn_token_to_image.forward(&q, &k, &keys)?;
        let queries = (queries + attn_out)?.apply(&self.norm_final_attn)?;

        Ok((queries, keys))
    }
}

#[derive(Clone)]
pub struct RoPEAttention {
    attention: Attention,
    rope_theta: f32,
    rope_k_repeat: bool,
    feat_sizes: (usize, usize),
    freqs_cis: Tensor,
}

impl RoPEAttention {
    pub fn new(
        embedding_dim: usize,
        num_heads: usize,
        downsample_rate: usize,
        rope_theta: f32,
        rope_k_repeat: bool,
        feat_sizes: (usize, usize),
        kv_in_dim: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let device = vb.device().clone();
        let attention = Attention::new(embedding_dim, num_heads, downsample_rate, kv_in_dim, vb)?;

        let freqs_cis = compute_axial_cis(
            attention.internal_dim() / num_heads,
            feat_sizes.0,
            feat_sizes.1,
            rope_theta,
            &device,
        )?;

        Ok(Self {
            attention,
            freqs_cis,
            rope_theta,
            rope_k_repeat,
            feat_sizes,
        })
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        num_k_exclude_rope: usize,
    ) -> Result<Tensor> {

        // 投影和分头
        let q_proj = self.attention.q_proj.forward(q)?;
        let k_proj = self.attention.k_proj.forward(k)?;
        let v_proj = self.attention.v_proj.forward(v)?;
        let q = self.attention.separate_heads(&q_proj)?;
        let k = self.attention.separate_heads(&k_proj)?;
        let v = self.attention.separate_heads(&v_proj)?;

        // 扩展序列长度以匹配k
        /*
        if self.rope_k_repeat {
            let seq_len_k = k.dim(2)?;
            let repeat_factor = seq_len_k / (w * h);
            freqs_cis = freqs_cis.unsqueeze(0)?.unsqueeze(0)?.repeat((1, 1, repeat_factor, 1))?;
        }*/

        // 应用旋转编码
        let (q, k) = apply_rotary_to_qk(
            &q,
            &k,
            num_k_exclude_rope,
            &self.freqs_cis,
            self.rope_k_repeat,
        )?;

        // 计算注意力
        //let scale = (self.attention.internal_dim as f64 / self.attention.num_heads as f64).sqrt();
        //let attn = (q.matmul(&k.transpose(2, 3)?)? / scale)?;
        //let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        //let out = attn.matmul(&v)?;
        let out = scaled_dot_product_attention(&q, &k, &v)?;

        let out = self.attention.recombine_heads(&out)?;
        let res = self.attention.out_proj.forward(&out)?;

        Ok(res)
        //self.attention.out_proj.forward(&out)
    }
}

fn init_t_xy(end_x: usize, end_y: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let prod = end_x * end_y;
    let t = Tensor::arange(0u32, prod as u32, device)?.to_dtype(DType::F32)?;

    // mod op：t_x = t - (t // end_x) * end_x
    let end_x_tensor = Tensor::new(end_x as f32, device)?;
    let quotient = t.broadcast_div(&end_x_tensor)?.floor()?;
    let t_x = t.broadcast_sub(&quotient.broadcast_mul(&end_x_tensor)?)?;

    // t_y = floor(t / end_x)
    let t_y = quotient;

    Ok((t_x, t_y))
}

pub fn compute_axial_cis(
    dim: usize,
    end_x: usize,
    end_y: usize,
    theta: f32,
    device: &Device,
) -> Result<Tensor> {
    let k = dim / 4;
    let indices = Tensor::arange_step(0u32, dim as u32, 4, device)?
        .to_dtype(DType::F32)?
        .narrow(0, 0, k)?;

    let exponent = indices.broadcast_div(&Tensor::new(dim as f32, device)?)?;
    let freqs_base = Tensor::new(theta, device)?
        .broadcast_pow(&exponent)?
        .recip()?;
    let (freqs_x, freqs_y) = (freqs_base.clone(), freqs_base);

    let (t_x, t_y) = init_t_xy(end_x, end_y, device)?;
    let prod = end_x * end_y;
    let t_x = t_x.reshape((prod, 1))?;
    let t_y = t_y.reshape((prod, 1))?;

    let freqs_x = t_x.broadcast_mul(&freqs_x.reshape((1, k))?)?; // (prod, k)
    let freqs_y = t_y.broadcast_mul(&freqs_y.reshape((1, k))?)?;

    let real_x = freqs_x.cos()?.unsqueeze(2)?; // (prod, k, 1)
    let imag_x = freqs_x.sin()?.unsqueeze(2)?;
    let real_y = freqs_y.cos()?.unsqueeze(2)?;
    let imag_y = freqs_y.sin()?.unsqueeze(2)?;

    let x_complex = Tensor::cat(&[real_x, imag_x], 2)?; // (prod, k, 2)
    let y_complex = Tensor::cat(&[real_y, imag_y], 2)?;
    let combined = Tensor::cat(&[x_complex, y_complex], 1)?; // (prod, 2k, 2)

    Ok(combined)
}


fn reshape_for_broadcast(freqs_cis: &Tensor, x: &Tensor) -> Result<Tensor> {
    let x_dims = x.dims();
    let freqs_dims = freqs_cis.dims();

    // 确保 freqs_cis 的最后两维与 x 的最后两维匹配
    if freqs_dims[freqs_dims.len() - 2] != x_dims[x_dims.len() - 2]
        || freqs_dims[freqs_dims.len() - 1] != x_dims[x_dims.len() - 1]
    {
        return Err(candle_core::Error::Msg(
            "Last two dimensions of freqs_cis and x must match".into(),
        ));
    }

    // 构建新的形状：前面补1直到维度数与x相同，最后两维保留
    let mut new_shape = Vec::with_capacity(x_dims.len());
    for _ in 0..(x_dims.len() - freqs_dims.len()) {
        new_shape.push(1);
    }
    new_shape.extend_from_slice(freqs_dims);

    freqs_cis.reshape(new_shape)
}

pub fn apply_rotary_enc(
    xq: &Tensor,
    xk: &Tensor,
    freqs_cis: &Tensor,
    repeat_freqs_k: bool,
) -> Result<(Tensor, Tensor)> {
    // 将输入拆分为实部和虚部
    // 处理xq的reshape
    let xq_dims = xq.dims();
    let mut new_xq_shape = xq_dims[..xq_dims.len() - 1].to_vec();
    let orig_last_dim = xq_dims.last().unwrap();
    new_xq_shape.push(orig_last_dim / 2); // 原最后一个维度减半
    new_xq_shape.push(2); // 添加复数维度
    let xq = xq.reshape(new_xq_shape.as_slice())?;

    // 对xk执行相同操作
    let xk_dims = xk.dims();
    let mut new_xk_shape = xk_dims[..xk_dims.len() - 1].to_vec();
    let orig_last_dim = xk_dims.last().unwrap();
    new_xk_shape.push(orig_last_dim / 2);
    new_xk_shape.push(2);
    let xk = xk.reshape(new_xk_shape.as_slice())?;

    // 调整频率张量的形状以进行广播
    let freqs_cis = reshape_for_broadcast(freqs_cis, &xq)?;

    // 分解实部和虚部
    let xq_real = xq.i((.., .., .., .., 0))?;
    let xq_imag = xq.i((.., .., .., .., 1))?;
    let freqs_real = freqs_cis.i((.., .., .., .., 0))?;
    let freqs_imag = freqs_cis.i((.., .., .., .., 1))?;

    // 复数乘法
    let xq_out_real =
        (xq_real.broadcast_mul(&freqs_real)? - xq_imag.broadcast_mul(&freqs_imag)?)?;
    let xq_out_imag =
        (xq_real.broadcast_mul(&freqs_imag)? + xq_imag.broadcast_mul(&freqs_real)?)?;

    // 处理xk
    //let xk_real = xk.i((.., .., 0))?;
    //let xk_imag = xk.i((.., .., 1))?;
    let xk_real = xk.i((.., .., .., .., 0))?;
    let xk_imag = xk.i((.., .., .., .., 1))?;

    let (xk_out_real, xk_out_imag) = if repeat_freqs_k {
        // 扩展频率张量以匹配xk的序列长度
        let seq_len = xk.dims()[xk.dims().len() - 3];
        let r = seq_len / freqs_cis.dims()[freqs_cis.dims().len() - 3];

        let freqs_real = freqs_real.repeat((1, 1, r, 1))?;
        let freqs_imag = freqs_imag.repeat((1, 1, r, 1))?;
        (
            (xk_real.broadcast_mul(&freqs_real)? - xk_imag.broadcast_mul(&freqs_imag)?)?,
            (xk_real.broadcast_mul(&freqs_imag)? + xk_imag.broadcast_mul(&freqs_real)?)?,
        )
    } else {
        (
            (xk_real.broadcast_mul(&freqs_real)? - xk_imag.broadcast_mul(&freqs_imag)?)?,
            (xk_real.broadcast_mul(&freqs_imag)? + xk_imag.broadcast_mul(&freqs_real)?)?,
        )
    };


    //view_as_real?;
    let xq_out_ = Tensor::cat(
        &[
            xq_out_real.unsqueeze(D::Minus1)?,
            xq_out_imag.unsqueeze(D::Minus1)?,
        ],
        D::Minus1,
    )?;
    let xq_out = xq_out_.flatten(3, 4)?; // 将 [..., 128, 2] → [..., 256]

    // 对xk执行相同操作
    let xk_out = Tensor::cat(
        &[
            xk_out_real.unsqueeze(D::Minus1)?,
            xk_out_imag.unsqueeze(D::Minus1)?,
        ],
        D::Minus1,
    )?
    .flatten(3, 4)?;

    Ok((xq_out, xk_out))
}

pub fn apply_rotary_to_qk(
    q: &Tensor,
    k: &Tensor,
    num_k_exclude_rope: usize,
    freqs_cis: &Tensor,
    rope_k_repeat: bool,
) -> Result<(Tensor, Tensor)> {
    // 确定需要应用旋转编码的k头数
    let k_heads = k.dim(2)?; // 获取k的第三个维度大小（heads数）
    let num_k_rope = k_heads - num_k_exclude_rope;

    // 对k进行切片操作 [1, 1, 4100, 256] → 取前4096个head
    let k_rope_part = k.narrow(2, 0, num_k_rope)?; // [1, 1, 4096, 256]

    // 应用旋转位置编码
    let (q_rotated, k_rotated_part) = apply_rotary_enc(q, &k_rope_part, freqs_cis, rope_k_repeat)?;

    // 拼接处理后的k部分和未处理的剩余部分
    let k_out = if num_k_exclude_rope > 0 {
        let k_remain = k.narrow(2, num_k_rope, num_k_exclude_rope)?; // [1, 1, 4, 256]
        Tensor::cat(&[&k_rotated_part, &k_remain], 2)? // 沿第3维拼接
    } else {
        k_rotated_part
    };

    // 验证最终形状
    //debug_assert_eq!(q_rotated.dims(), &[1, 1, 4096, 256]);
    //debug_assert_eq!(k_out.dims(), &[1, 1, 4100, 256]);

    Ok((q_rotated, k_out))
}
