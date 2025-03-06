use candle_core::{DType, IndexOp, Result, Tensor};
use candle_nn::{Activation, Conv2d, ConvTranspose2dConfig, Embedding, Module, VarBuilder};

use crate::modeling::sam::transformer::TwoWayTransformer;
use crate::modeling::sam_utils::{LayerNorm2d, MLP};

pub struct MaskDecoder {
    // Embeddings
    transformer_dim: usize,
    iou_token: Embedding,
    mask_tokens: Embedding,
    obj_score_token: Option<Embedding>,

    // Network components
    output_upscaling: Vec<Box<dyn Module>>,
    conv_s0_s1: Option<(Conv2d, Conv2d)>,
    output_hypernetworks_mlps: Vec<MLP>,
    iou_prediction_head: MLP,
    pred_obj_score_head: Option<MLP>,
    transformer: TwoWayTransformer,

    // Config parameters
    num_mask_tokens: usize,
    use_high_res_features: bool,
    //iou_prediction_use_sigmoid: bool,
    dynamic_multimask_via_stability: bool,
    dynamic_multimask_stability_delta: f32,
    dynamic_multimask_stability_thresh: f32,
}

impl MaskDecoder {
    pub fn new(
        transformer_dim: usize,
        transformer: TwoWayTransformer,
        num_multimask_outputs: usize,
        iou_head_depth: usize,
        iou_head_hidden_dim: usize,
        use_high_res_features: bool,
        iou_prediction_use_sigmoid: bool,
        dynamic_multimask_via_stability: bool,
        dynamic_multimask_stability_delta: f32,
        dynamic_multimask_stability_thresh: f32,
        pred_obj_scores: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let num_mask_tokens = num_multimask_outputs + 1;

        // Initialize embeddings
        let iou_token = candle_nn::embedding(1, transformer_dim, vb.pp("iou_token"))?;
        let mask_tokens =
            candle_nn::embedding(num_mask_tokens, transformer_dim, vb.pp("mask_tokens"))?;
        let obj_score_token = if pred_obj_scores {
            Some(candle_nn::embedding(
                1,
                transformer_dim,
                vb.pp("obj_score_token"),
            )?)
        } else {
            None
        };
        // all sam config use_multimask_token_for_obj_ptr is True

        // Initialize upscaling components
        let mut output_upscaling: Vec<Box<dyn Module>> = Vec::new();
        output_upscaling.push(Box::new(candle_nn::conv_transpose2d(
            transformer_dim,
            transformer_dim / 4,
            2,
            ConvTranspose2dConfig {
                stride: 2,          
                ..Default::default()
            },
            vb.pp("output_upscaling.0"),
        )?));
        output_upscaling.push(Box::new(LayerNorm2d::new(
            transformer_dim / 4,
            1e-6,
            vb.pp("output_upscaling.1"),
        )?));
        output_upscaling.push(Box::new(Activation::Gelu));
        output_upscaling.push(Box::new(candle_nn::conv_transpose2d(
            transformer_dim / 4,
            transformer_dim / 8,
            2,
            ConvTranspose2dConfig {
                stride: 2,
                ..Default::default()
            },
            vb.pp("output_upscaling.3"),
        )?));
        output_upscaling.push(Box::new(Activation::Gelu));

        // Initialize high-res features components if needed
        let conv_s0_s1 = if use_high_res_features {
            Some((
                candle_nn::conv2d(
                    transformer_dim,
                    transformer_dim / 8,
                    1,
                    Default::default(),
                    vb.pp("conv_s0"),
                )?,
                candle_nn::conv2d(
                    transformer_dim,
                    transformer_dim / 4,
                    1,
                    Default::default(),
                    vb.pp("conv_s1"),
                )?,
            ))
        } else {
            None
        };

        // Initialize MLPs
        let mut output_hypernetworks_mlps = Vec::new();
        for i in 0..num_mask_tokens {
            let mlp = MLP::new(
                vb.pp(&format!("output_hypernetworks_mlps.{}", i)),
                transformer_dim,
                transformer_dim,
                transformer_dim / 8,
                3,
                Activation::Relu,
                false,
            )?;
            output_hypernetworks_mlps.push(mlp);
        }

        // Initialize prediction heads
        let iou_prediction_head = MLP::new(
            vb.pp("iou_prediction_head"),
            transformer_dim,
            iou_head_hidden_dim,
            num_mask_tokens,
            iou_head_depth,
            Activation::Relu,
            iou_prediction_use_sigmoid,
        )?;

        let pred_obj_score_head = if pred_obj_scores {
            Some(MLP::new(
                vb.pp("pred_obj_score_head"),
                transformer_dim,
                transformer_dim,
                1,
                3,
                Activation::Relu,
                false,
            )?)
        } else {
            None
        };

        Ok(Self {
            transformer_dim,
            iou_token,
            mask_tokens,
            obj_score_token,
            output_upscaling,
            conv_s0_s1,
            output_hypernetworks_mlps,
            iou_prediction_head,
            pred_obj_score_head,
            transformer,
            num_mask_tokens,
            use_high_res_features,
            dynamic_multimask_via_stability,
            dynamic_multimask_stability_delta,
            dynamic_multimask_stability_thresh,
        })
    }

    pub fn forward_conv_s0_s1(
        &self,
        backbone_fpn_layer0: &Tensor,
        backbone_fpn_layer1: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if self.conv_s0_s1.is_none() {
            return Err(candle_core::Error::Msg(format!(
                "conv_s0 and conv_s1 not set!"
            )));
        }
        let (conv_s0, conv_s1) = self.conv_s0_s1.as_ref().unwrap();

        Ok((
            conv_s0.forward(backbone_fpn_layer0)?,
            conv_s1.forward(backbone_fpn_layer1)?,
        ))
    }

    fn predict_masks(
        &self,
        image_embeddings: &Tensor,
        image_pe: &Tensor,
        sparse_prompt_embeddings: &Tensor,
        dense_prompt_embeddings: &Tensor,
        high_res_features: Option<&(Tensor, Tensor)>,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        // Concatenate output tokens
        let mut tokens = Vec::new();
        let mut s = 0;

        if let Some(obj_token) = &self.obj_score_token {
            tokens.push(obj_token.embeddings());
            s += 1;
        }

        tokens.push(self.iou_token.embeddings());
        tokens.push(self.mask_tokens.embeddings());

        let output_tokens = Tensor::cat(&tokens, 0)?.unsqueeze(0)?;
        let output_tokens = output_tokens.broadcast_as((
            sparse_prompt_embeddings.dim(0)?,
            output_tokens.dim(1)?,
            output_tokens.dim(2)?,
        ))?;

        let tokens = Tensor::cat(&[&output_tokens, sparse_prompt_embeddings], 1)?;

        // Process image embeddings
        let src = image_embeddings.broadcast_add(dense_prompt_embeddings)?;
        let pos_src = image_pe.broadcast_as(src.shape())?;
        let src_shape = src.shape().clone();


        // Run transformer
        let (hs, src) = self
            .transformer
            .forward(&src, &pos_src, &tokens.contiguous()?)?;

        // Process outputs
        let iou_token_out = hs.i((.., s))?;
        let mask_start = s + 1;
        let mask_end = mask_start + self.num_mask_tokens;
        let mask_tokens_out = hs.i((.., mask_start..mask_end))?;

        let src = src.transpose(1, 2)?.reshape(src_shape)?;
        // Upscale embeddings
        let upscaled_embedding = if self.use_high_res_features {
            let (conv_s0, conv_s1) = self.conv_s0_s1.as_ref().unwrap();
            let (feat_s0, feat_s1) = high_res_features.unwrap();
            let x = self.output_upscaling[0].forward(&src)?;
            let x = self.output_upscaling[1].forward(&(x + feat_s1)?)?;
            let x = self.output_upscaling[2].forward(&x)?;
            let x = self.output_upscaling[3].forward(&x)?;

            self.output_upscaling[4].forward(&(x + feat_s0)?)?
        } else {
            self.output_upscaling
                .iter()
                .try_fold(src.clone(), |x, m| m.forward(&x))?
        };

        // Generate masks
        let hyper_in: Vec<Tensor> = self
            .output_hypernetworks_mlps
            .iter()
            .enumerate()
            .map(|(i, mlp)| mlp.forward(&mask_tokens_out.i((.., i))?))
            .collect::<Result<_>>()?;
        let hyper_in = Tensor::stack(&hyper_in, 1)?;

        let (b, c, h, w) = upscaled_embedding.dims4()?;
        let masks = hyper_in
            .matmul(&upscaled_embedding.reshape((b, c, h * w))?)?
            .reshape((b, self.num_mask_tokens, h, w))?;

        // Generate predictions
        let iou_pred = self.iou_prediction_head.forward(&iou_token_out)?;
        let object_score_logits = if let Some(head) = &self.pred_obj_score_head {
            head.forward(&hs.i((.., 0))?)?
        } else {
            Tensor::ones((iou_pred.dim(0)?, 1), DType::F32, iou_pred.device())?
                .broadcast_mul(&Tensor::new(10f32, iou_pred.device())?)?
        };

        Ok((masks, iou_pred, mask_tokens_out, object_score_logits))
    }

    fn _get_stability_scores(&self, mask_logits: &Tensor) -> Result<Tensor> {
        let mask_logits_flat = mask_logits.flatten_from(2)?;
        let stability_delta = self.dynamic_multimask_stability_delta;

        let area_i = mask_logits_flat
            .ge(stability_delta)?
            .to_dtype(DType::F32)?
            .sum(2)?;

        let area_u = mask_logits_flat
            .ge(-stability_delta)?
            .to_dtype(DType::F32)?
            .sum(2)?;

        let ratio = area_i.div(&area_u)?;
        let area_u_nonzero = area_u.gt(0.0)?;

        let stability_scores = Tensor::where_cond(
            &area_u_nonzero,
            &ratio,
            &Tensor::ones_like(&area_i)?,
        )?;

        Ok(stability_scores)
    }

    fn _dynamic_multimask_via_stability(
        &self,
        all_mask_logits: &Tensor,
        all_iou_scores: &Tensor,
    ) -> Result<(Tensor, Tensor)> {

        let multimask_logits = all_mask_logits.narrow(1, 1, 3)?;
        let multimask_iou_scores = all_iou_scores.narrow(1, 1, 3)?;

        // find beat score index
        let best_scores_inds = multimask_iou_scores.argmax(1)?;

        //let batch_size = multimask_iou_scores.dim(0)?;
        //let batch_inds = Tensor::arange(0u32, batch_size as u32, all_iou_scores.device())?;
        //let idx = best_scores_inds.unsqueeze(1)?.unsqueeze(2)?;

        let best_multimask_logits = multimask_logits
            .index_select(&best_scores_inds, 1)?
            .squeeze(1)?;

        let best_multimask_iou_scores = multimask_iou_scores
            .index_select(&best_scores_inds, 1)?
            .squeeze(1)?;

        // get token 0 output
        let singlemask_logits = all_mask_logits.narrow(1, 0, 1)?;
        let singlemask_iou_scores = all_iou_scores.narrow(1, 0, 1)?;


        let stability_scores = self._get_stability_scores(&singlemask_logits)?;
        let is_stable = stability_scores.ge(self.dynamic_multimask_stability_thresh)?;

        // expend to [B, 1, H, W]）
        let is_stable_3d = is_stable
            .unsqueeze(1)? // add channel
            .unsqueeze(2)? // add height
            .broadcast_as(singlemask_logits.shape())?;

        let mask_logits_out = Tensor::where_cond(
            &is_stable_3d,
            &singlemask_logits,
            &best_multimask_logits.unsqueeze(1)?, // 添加通道维度
        )?;

        let iou_scores_out = Tensor::where_cond(
            &is_stable.broadcast_as(singlemask_iou_scores.shape())?,
            &singlemask_iou_scores,
            &best_multimask_iou_scores.unsqueeze(1)?,
        )?;

        Ok((mask_logits_out, iou_scores_out))
    }

    pub fn forward(
        &self,
        image_embeddings: &Tensor,
        image_pe: &Tensor,
        sparse_prompt_embeddings: &Tensor,
        dense_prompt_embeddings: &Tensor,
        multimask_output: bool,
        high_res_features: Option<&(Tensor, Tensor)>,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (all_masks, all_iou, mask_tokens_out, obj_score) = self.predict_masks(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            high_res_features,
        )?;


        let (masks, iou_pred) = if multimask_output {
            (all_masks.i((.., 1..))?, all_iou.i((.., 1..))?)
        } else if self.dynamic_multimask_via_stability {
            self._dynamic_multimask_via_stability(&all_masks, &all_iou)?
        } else {
            (all_masks.i((.., 0..1))?, all_iou.i((.., 0..1))?)
        };

        let sam_tokens = if multimask_output {
            mask_tokens_out.i((.., 1..))?
        } else {
            mask_tokens_out.i((.., 0..1))?
        };

        Ok((masks, iou_pred, sam_tokens, obj_score))
    }
}
