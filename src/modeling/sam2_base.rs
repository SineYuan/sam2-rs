use std::collections::{BTreeMap, HashMap};

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Module, VarBuilder};

use crate::modeling::interpolation::*;
use crate::modeling::sam::transformer::TwoWayTransformer;
use crate::modeling::sam::{mask_decoder::MaskDecoder, prompt_encoder::PromptEncoder};
use crate::modeling::sam_utils::{Identity, MLP};
use crate::modeling::{
    backbones::{
        hieradet::{Hiera, HieraConfig},
        image_encoder::{FpnNeck, ImageEncoder},
    },
    memory_attention::*,
    memory_encoder::*,
    position_encoding::*,
};

pub const NO_OBJ_SCORE: f32 = -1024.0;

macro_rules! get_scalar {
    ($ty:ty, $tensors:expr, $key:expr, $default:expr) => {
        $tensors
            .get($key)
            .and_then(|t| t.to_scalar::<$ty>().ok())
            .unwrap_or($default)
    };
}

macro_rules! get_bool {
    ($tensors:expr, $key:expr, $default:expr) => {
        $tensors
            .get($key)
            .and_then(|t| t.to_scalar::<u8>().ok())
            .map(|v| v > 0)
            .unwrap_or($default)
    };
}

macro_rules! get_vec {
    ($ty:ty, $tensors:expr, $key:expr, $default:expr) => {
        $tensors
            .get($key)
            .and_then(|t| t.to_vec1::<$ty>().ok())
            .unwrap_or_else(|| $default.clone())
    };
}

macro_rules! get_scalar_opt {
    ($ty:ty, $tensors:expr, $key:expr) => {
        $tensors
            .get($key)
            .and_then(|t| t.to_scalar::<$ty>().ok())
            .map(|v| v as usize)
    };
}

#[derive(Debug)]
pub enum Prompt {
    Point(u32, u32, u8),     // (x, y, label)
    Box(u32, u32, u32, u32), // (x1, y1, x2, y2),
}

pub struct TrackState {
    pub frame_outputs: HashMap<i32, FrameOutput>,
    pub current_memory: Option<Tensor>,
}

#[derive(Clone, Debug)]
pub struct FrameOutput {
    pub pred_masks: Tensor,                         
    pub pred_masks_high_res: Tensor,                
    pub obj_ptr: Tensor,                            
    pub object_score_logits: Tensor,                
    pub maskmem_features: Option<(Tensor, Tensor)>, 
}

pub struct SAM2Base {
    device: Device,

    image_encoder: ImageEncoder,
    memory_attention: MemoryAttention,
    memory_encoder: MemoryEncoder,
    sam_prompt_encoder: PromptEncoder,
    sam_mask_decoder: MaskDecoder,

    obj_ptr_proj: Box<dyn Module>,
    obj_ptr_tpos_proj: Box<dyn Module>,

    maskmem_tpos_enc: Tensor,
    image_size: usize,
    hidden_dim: usize,
    mem_dim: usize,
    no_mem_embed: Tensor,
    no_mem_pos_enc: Tensor,
    no_obj_ptr: Option<Tensor>,
    no_obj_embed_spatial: Option<Tensor>,

    use_high_res_features_in_sam: bool,
    num_feature_levels: usize,
    num_maskmem: usize,

    use_mask_input_as_output_without_sam: bool,
    max_cond_frames_in_attn: usize,

    directly_add_no_mem_embed: bool,

    ext_config: SAM2BaseExtConfig,
}

pub struct SAM2BaseExtConfig {
    backbone_stride: usize,

    multimask_min_pt_num: usize,
    multimask_max_pt_num: usize,
    multimask_output_for_tracking: bool,
    multimask_output_in_sam: bool,

    pred_obj_scores: bool,
    soft_no_obj_ptr: bool,
    fixed_no_obj_ptr: bool,

    use_mlp_for_obj_ptr_proj: bool,
    use_obj_ptrs_in_encoder: bool,
    max_obj_ptrs_in_encoder: usize,

    sigmoid_scale_for_mem_enc: f32,
    sigmoid_bias_for_mem_enc: f32,
    binarize_mask_from_pts_for_mem_enc: bool,

    memory_temporal_stride_for_eval: usize,
    proj_tpos_enc_in_obj_ptrs: bool,
    use_signed_tpos_enc_to_obj_ptrs: bool,
    only_obj_ptrs_in_the_past_for_eval: bool,

    dynamic_multimask_via_stability: bool,
    dynamic_multimask_stability_delta: f32,
    dynamic_multimask_stability_thresh: f32,
}

impl Default for SAM2BaseExtConfig {
    fn default() -> Self {
        Self {
            backbone_stride: 16,
            multimask_min_pt_num: 1,
            multimask_max_pt_num: 1,
            multimask_output_for_tracking: false,
            multimask_output_in_sam: false,
            pred_obj_scores: false,
            soft_no_obj_ptr: false,
            fixed_no_obj_ptr: false,
            use_mlp_for_obj_ptr_proj: false,
            use_obj_ptrs_in_encoder: false,
            max_obj_ptrs_in_encoder: 16,
            sigmoid_scale_for_mem_enc: 1.0,
            sigmoid_bias_for_mem_enc: 0.0,
            binarize_mask_from_pts_for_mem_enc: false,
            memory_temporal_stride_for_eval: 1,
            proj_tpos_enc_in_obj_ptrs: true,
            use_signed_tpos_enc_to_obj_ptrs: false,
            only_obj_ptrs_in_the_past_for_eval: false,

            dynamic_multimask_via_stability: true,
            dynamic_multimask_stability_delta: 0.05,
            dynamic_multimask_stability_thresh: 0.98,
        }
    }
}

impl SAM2Base {
    pub fn new(
        vb: VarBuilder,
        image_encoder: ImageEncoder,
        memory_attention: MemoryAttention,
        memory_encoder: MemoryEncoder,

        // 配置参数
        num_maskmem: usize,
        image_size: usize,
        use_high_res_features_in_sam: bool,
        use_obj_ptrs_in_encoder: bool,
        proj_tpos_enc_in_obj_ptrs: bool,
        pred_obj_scores: bool,
        directly_add_no_mem_embed: bool,
        iou_prediction_use_sigmoid: bool,

        use_mask_input_as_output_without_sam: bool,
        max_cond_frames_in_attn: usize,

        ext_config: SAM2BaseExtConfig,
    ) -> Result<Self> {
        let hidden_dim = image_encoder.neck_d_model();
        let mem_dim = memory_encoder.output_dim();
        let num_feature_levels = if use_high_res_features_in_sam {
            3usize
        } else {
            1usize
        };

        let no_mem_embed = vb.get((1, 1, hidden_dim), "no_mem_embed")?;

        let no_mem_pos_enc = vb.get((1, 1, hidden_dim), "no_mem_pos_enc")?;

        let maskmem_tpos_enc = vb.get((num_maskmem, 1, 1, mem_dim), "maskmem_tpos_enc")?;

        let sam_prompt_encoder = PromptEncoder::new(
            hidden_dim,
            (
                image_size / ext_config.backbone_stride,
                image_size / ext_config.backbone_stride,
            ),
            (image_size, image_size),
            16,
            vb.pp("sam_prompt_encoder"),
        )?;

        let sam_mask_decoder = MaskDecoder::new(
            hidden_dim,
            TwoWayTransformer::new(
                2,
                hidden_dim,
                8,
                2048,
                vb.pp("sam_mask_decoder.transformer"),
            )?,
            3, // num_multimask_outputs
            3,
            256,
            use_high_res_features_in_sam,
            iou_prediction_use_sigmoid,
            ext_config.dynamic_multimask_via_stability,
            ext_config.dynamic_multimask_stability_delta,
            ext_config.dynamic_multimask_stability_thresh,
            pred_obj_scores,
            vb.pp("sam_mask_decoder"),
        )?;

        // 初始化其他参数
        let no_obj_ptr = if pred_obj_scores && use_obj_ptrs_in_encoder {
            Some(vb.get((1, hidden_dim), "no_obj_ptr")?)
        } else {
            None
        };

        let obj_ptr_proj: Box<dyn Module> = if use_obj_ptrs_in_encoder {
            if ext_config.use_mlp_for_obj_ptr_proj {
                Box::new(MLP::new(
                    vb.pp("obj_ptr_proj"),
                    hidden_dim,
                    hidden_dim,
                    hidden_dim,
                    3,
                    candle_nn::Activation::Relu,
                    false,
                )?)
            } else {
                Box::new(candle_nn::linear(
                    hidden_dim,
                    hidden_dim,
                    vb.pp("obj_ptr_proj"),
                )?)
            }
        } else {
            Box::new(Identity::new())
        };

        let obj_ptr_tpos_proj: Box<dyn Module> = if proj_tpos_enc_in_obj_ptrs {
            Box::new(candle_nn::linear(
                hidden_dim,
                mem_dim,
                vb.pp("obj_ptr_tpos_proj"),
            )?)
        } else {
            Box::new(Identity::new())
        };

        Ok(Self {
            image_encoder,
            memory_attention,
            memory_encoder,
            sam_prompt_encoder,
            device: vb.device().clone(),
            sam_mask_decoder,
            maskmem_tpos_enc,
            image_size,
            hidden_dim,
            mem_dim,
            no_mem_embed,
            no_mem_pos_enc,
            no_obj_ptr,
            no_obj_embed_spatial: None,
            use_high_res_features_in_sam,
            directly_add_no_mem_embed,
            num_feature_levels,
            num_maskmem,
            obj_ptr_proj,
            obj_ptr_tpos_proj,
            use_mask_input_as_output_without_sam,
            max_cond_frames_in_attn,

            ext_config: ext_config,
        })
    }

    pub fn from_safetensors<P: AsRef<std::path::Path>>(
        checkpoint_path: P,
        device: &Device,
    ) -> Result<Self> {
        let mut chkp_tensors = candle_core::safetensors::load(checkpoint_path, device)?;
        let prefix = "config";

        // === 第一阶段：参数读取 ===
        // 基础参数

        let scalp = get_scalar!(
            i64,
            &chkp_tensors,
            &format!("{}.image_encoder.scalp", prefix),
            1
        ) as usize;

        let image_size =
            get_scalar!(i64, &chkp_tensors, &format!("{}.image_size", prefix), 512) as usize;
        let num_maskmem =
            get_scalar!(i64, &chkp_tensors, &format!("{}.num_maskmem", prefix), 1) as usize;
        let use_high_res_features_in_sam = get_bool!(
            &chkp_tensors,
            &format!("{}.use_high_res_features_in_sam", prefix),
            false
        );
        let use_obj_ptrs_in_encoder = get_bool!(
            &chkp_tensors,
            &format!("{}.use_obj_ptrs_in_encoder", prefix),
            true
        );
        let proj_tpos_enc_in_obj_ptrs = get_bool!(
            &chkp_tensors,
            &format!("{}.proj_tpos_enc_in_obj_ptrs", prefix),
            true
        );
        let pred_obj_scores =
            get_bool!(&chkp_tensors, &format!("{}.pred_obj_scores", prefix), true);
        let directly_add_no_mem_embed = get_bool!(
            &chkp_tensors,
            &format!("{}.directly_add_no_mem_embed", prefix),
            true
        );
        let iou_prediction_use_sigmoid = get_bool!(
            &chkp_tensors,
            &format!("{}.iou_prediction_use_sigmoid", prefix),
            true
        );
        let use_mask_input_as_output_without_sam = get_bool!(
            &chkp_tensors,
            &format!("{}.use_mask_input_as_output_without_sam", prefix),
            false
        );
        let max_cond_frames_in_attn = get_scalar!(
            i64,
            &chkp_tensors,
            &format!("{}.max_cond_frames_in_attn", prefix),
            1
        ) as usize;
        let multimask_output_for_tracking = get_bool!(
            &chkp_tensors,
            &format!("{}.multimask_output_for_tracking", prefix),
            false
        );
        let multimask_max_pt_num = get_scalar!(
            i64,
            &chkp_tensors,
            &format!("{}.multimask_max_pt_num", prefix),
            1
        ) as usize;
        let multimask_min_pt_num = get_scalar!(
            i64,
            &chkp_tensors,
            &format!("{}.multimask_min_pt_num", prefix),
            1
        ) as usize;
        let multimask_output_in_sam = get_bool!(
            &chkp_tensors,
            &format!("{}.multimask_output_in_sam", prefix),
            false
        );
        let pred_obj_scores =
            get_bool!(&chkp_tensors, &format!("{}.pred_obj_scores", prefix), false);
        let use_mlp_for_obj_ptr_proj = get_bool!(
            &chkp_tensors,
            &format!("{}.use_mlp_for_obj_ptr_proj", prefix),
            false
        );
        let fixed_no_obj_ptr = get_bool!(
            &chkp_tensors,
            &format!("{}.fixed_no_obj_ptr", prefix),
            false
        );
        let sigmoid_scale_for_mem_enc = get_scalar!(
            f32,
            &chkp_tensors,
            &format!("{}.sigmoid_scale_for_mem_enc", prefix),
            0.0f32
        ) as f32;
        let sigmoid_bias_for_mem_enc = get_scalar!(
            f32,
            &chkp_tensors,
            &format!("{}.sigmoid_bias_for_mem_enc", prefix),
            0.0f32
        ) as f32;
        let memory_temporal_stride_for_eval = get_scalar!(
            i64,
            &chkp_tensors,
            &format!("{}.memory_temporal_stride_for_eval", prefix),
            1
        ) as usize;
        // max_obj_ptrs_in_encoder
        let use_signed_tpos_enc_to_obj_ptrs = get_bool!(
            &chkp_tensors,
            &format!("{}.use_signed_tpos_enc_to_obj_ptrs", prefix),
            false
        );
        let only_obj_ptrs_in_the_past_for_eval = get_bool!(
            &chkp_tensors,
            &format!("{}.only_obj_ptrs_in_the_past_for_eval", prefix),
            false
        );

        let mut ext_config = SAM2BaseExtConfig::default();
        ext_config.multimask_output_for_tracking = multimask_output_for_tracking;
        ext_config.multimask_min_pt_num = multimask_min_pt_num;
        ext_config.multimask_max_pt_num = multimask_max_pt_num;
        ext_config.multimask_output_in_sam = multimask_output_in_sam;
        ext_config.pred_obj_scores = pred_obj_scores;
        ext_config.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj;
        ext_config.fixed_no_obj_ptr = fixed_no_obj_ptr;
        ext_config.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc;
        ext_config.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc;
        ext_config.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval;
        ext_config.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder;
        ext_config.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs;
        ext_config.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval;

        // ImageEncoder参数
        let pe_prefix = format!("{}.image_encoder.neck.position_encoding", prefix);
        let pe_params = (
            get_scalar!(
                i64,
                &chkp_tensors,
                &format!("{}.num_pos_feats", pe_prefix),
                256
            ) as usize,
            get_scalar!(
                i64,
                &chkp_tensors,
                &format!("{}.temperature", pe_prefix),
                10000
            ) as i32,
            get_bool!(&chkp_tensors, &format!("{}.normalize", pe_prefix), true),
        );

        // FPN参数
        let fpn_prefix = format!("{}.image_encoder.neck", prefix);
        let fpn_params = (
            get_scalar!(i64, &chkp_tensors, &format!("{}.d_model", fpn_prefix), 256) as usize,
            get_vec!(
                i64,
                &chkp_tensors,
                &format!("{}.backbone_channel_list", fpn_prefix),
                vec![1152, 576, 288, 144]
            )
            .into_iter()
            .map(|v| v as usize)
            .collect::<Vec<_>>(),
            chkp_tensors
                .get(&format!("{}.fpn_top_down_levels", fpn_prefix))
                .and_then(|t| t.to_vec1::<i64>().ok())
                .map(|v| v.into_iter().map(|x| x as usize).collect()),
        );

        // Hiera
        let hiera_prefix = format!("{}.image_encoder.trunk", prefix);
        let mut hiera_conf = HieraConfig::default();
        if let Some(t) = chkp_tensors.get(&format!("{}.embed_dim", hiera_prefix)) {
            hiera_conf.embed_dim = t.to_scalar::<i64>()? as usize;
        }
        if let Some(t) = chkp_tensors.get(&format!("{}.num_heads", hiera_prefix)) {
            hiera_conf.num_heads = t.to_scalar::<i64>()? as usize;
        }
        if let Some(t) = chkp_tensors.get(&format!("{}.stages", hiera_prefix)) {
            hiera_conf.stages = t
                .to_vec1::<i64>()?
                .into_iter()
                .map(|v| v as usize)
                .collect();
        }
        if let Some(t) = chkp_tensors.get(&format!("{}.global_att_blocks", hiera_prefix)) {
            hiera_conf.global_att_blocks = t
                .to_vec1::<i64>()?
                .into_iter()
                .map(|v| v as usize)
                .collect();
        }
        if let Some(t) = chkp_tensors.get(&format!(
            "{}.window_pos_embed_bkg_spatial_size",
            hiera_prefix
        )) {
            let v = t.to_vec1::<i64>()?;
            hiera_conf.window_pos_embed_bkg_spatial_size = (v[0] as usize, v[1] as usize);
        }
        if let Some(t) = chkp_tensors.get(&format!("{}.window_spec", hiera_prefix)) {
            hiera_conf.window_spec = t
                .to_vec1::<i64>()?
                .into_iter()
                .map(|v| v as usize)
                .collect();
        }
        let image_size_pos_embed =
            chkp_tensors.remove(&format!("image_encoder.trunk.img_size_pos_embed"));

        // MemoryAttention
        let ma_prefix = format!("{}.memory_attention", prefix);
        let ma_params = {
            let parse_attention = |prefix: &str| -> Result<RoPEAttentionParams> {
                let feat_vec = get_vec!(
                    i64,
                    &chkp_tensors,
                    &format!("{}.feat_sizes", prefix),
                    vec![64, 64]
                )
                .into_iter()
                .map(|v| v as usize)
                .collect::<Vec<_>>();
                let feat_sizes = match feat_vec.as_slice() {
                    &[w, h] => (w, h),
                    _ => (64, 64),
                };

                let kv_in_dim = chkp_tensors
                    .get(&format!("{}.kv_in_dim", prefix))
                    .and_then(|t| t.to_scalar::<i64>().ok())
                    .map(|v| v as usize);

                Ok(RoPEAttentionParams {
                    embedding_dim: get_scalar!(
                        i64,
                        &chkp_tensors,
                        &format!("{}.embedding_dim", prefix),
                        256
                    ) as usize,
                    num_heads: get_scalar!(i64, &chkp_tensors, &format!("{}.num_heads", prefix), 8)
                        as usize,
                    downsample_rate: get_scalar!(
                        i64,
                        &chkp_tensors,
                        &format!("{}.downsample_rate", prefix),
                        1
                    ) as usize,
                    rope_theta: get_scalar!(
                        f64,
                        &chkp_tensors,
                        &format!("{}.rope_theta", prefix),
                        10000.0
                    ),
                    rope_k_repeat: get_bool!(
                        &chkp_tensors,
                        &format!("{}.rope_k_repeat", prefix),
                        false
                    ),
                    feat_sizes,
                    kv_in_dim: kv_in_dim,
                })
            };

            let mem_atten_layer_params = MemoryAttentionLayerParams {
                d_model: get_scalar!(
                    i64,
                    &chkp_tensors,
                    &format!("{}.layer.d_model", ma_prefix),
                    256
                ) as usize,
                dim_feedforward: get_scalar!(
                    i64,
                    &chkp_tensors,
                    &format!("{}.layer.dim_feedforward", ma_prefix),
                    2048
                ) as usize,
                dropout: get_scalar!(
                    f32,
                    &chkp_tensors,
                    &format!("{}.layer.dropout", ma_prefix),
                    0.0f32
                ) as f32,
                activation: Activation::Relu,
                self_attn: parse_attention(&format!("{}.layer.self_attention", ma_prefix))?,
                cross_attn_image: parse_attention(&format!("{}.layer.cross_attention", ma_prefix))?,
                pos_enc_at_attn: get_bool!(
                    &chkp_tensors,
                    &format!("{}.layer.pos_enc_at_attn", ma_prefix),
                    false
                ),
                pos_enc_at_cross_attn_queries: get_bool!(
                    &chkp_tensors,
                    &format!("{}.layer.pos_enc_at_cross_attn_queries", ma_prefix),
                    false
                ),
                pos_enc_at_cross_attn_keys: get_bool!(
                    &chkp_tensors,
                    &format!("{}.layer.pos_enc_at_cross_attn_keys", ma_prefix),
                    false
                ),
            };

            (
                get_scalar!(i64, &chkp_tensors, &format!("{}.d_model", ma_prefix), 256) as usize,
                get_bool!(
                    &chkp_tensors,
                    &format!("{}.pos_enc_at_input", ma_prefix),
                    true
                ),
                mem_atten_layer_params,
                get_scalar!(i64, &chkp_tensors, &format!("{}.num_layers", ma_prefix), 4) as usize,
                get_bool!(&chkp_tensors, &format!("{}.batch_first", ma_prefix), true),
            )
        };

        // MemoryEncoder
        let me_prefix = format!("{}.memory_encoder", prefix);
        let me_params = {
            let mask_prefix = format!("{}.mask_downsampler", me_prefix);
            let fuser_prefix = format!("{}.fuser", me_prefix);
            let pe_prefix = format!("{}.position_encoding", me_prefix);

            (
                // MaskDownSampler
                (
                    get_scalar!(
                        i64,
                        &chkp_tensors,
                        &format!("{}.kernel_size", mask_prefix),
                        3
                    ) as usize,
                    get_scalar!(i64, &chkp_tensors, &format!("{}.stride", mask_prefix), 2) as usize,
                    get_scalar!(i64, &chkp_tensors, &format!("{}.padding", mask_prefix), 1)
                        as usize,
                ),
                // Fuser
                (
                    get_scalar!(
                        i64,
                        &chkp_tensors,
                        &format!("{}.num_layers", fuser_prefix),
                        2
                    ) as usize,
                    CXBlockParams {
                        dim: get_scalar!(
                            i64,
                            &chkp_tensors,
                            &format!("{}.layer.dim", fuser_prefix),
                            256
                        ) as usize,
                        kernel_size: get_scalar!(
                            i64,
                            &chkp_tensors,
                            &format!("{}.layer.kernel_size", fuser_prefix),
                            3
                        ) as usize,
                        padding: get_scalar!(
                            i64,
                            &chkp_tensors,
                            &format!("{}.layer.padding", fuser_prefix),
                            1
                        ) as usize,
                        drop_path: 0.0,
                        layer_scale_init_value: get_scalar!(
                            f32,
                            &chkp_tensors,
                            &format!("{}.layer.padding", fuser_prefix),
                            10e-6
                        ) as f32,
                        use_dwconv: get_bool!(
                            &chkp_tensors,
                            &format!("{}.layer.padding", fuser_prefix),
                            true
                        ),
                    },
                ),
                // PositionEncoding
                (
                    get_scalar!(
                        i64,
                        &chkp_tensors,
                        &format!("{}.num_pos_feats", pe_prefix),
                        64
                    ) as usize,
                    get_scalar!(
                        i64,
                        &chkp_tensors,
                        &format!("{}.temperature", pe_prefix),
                        10000
                    ) as i32,
                    get_bool!(&chkp_tensors, &format!("{}.normalize", pe_prefix), true),
                ),
                get_scalar!(i64, &chkp_tensors, &format!("{}.in_dim", me_prefix), 256) as usize,
                get_scalar!(i64, &chkp_tensors, &format!("{}.out_dim", me_prefix), 256) as usize,
            )
        };

        // === build VarBuilder ===
        let vb = VarBuilder::from_tensors(chkp_tensors, DType::F32, device);

        // === init components ===
        // construct PositionEmbeddingSine
        let pe = PositionEmbeddingSine::new(
            pe_params.0,
            pe_params.1,
            pe_params.2,
            None,
            true,
            image_size,
            &[4, 8, 16, 32],
            device,
        );

        // construct FpnNeck
        let neck = FpnNeck::new(
            vb.pp("image_encoder.neck"),
            Box::new(pe),
            fpn_params.0,
            fpn_params.1,
            1,
            1,
            0, // kernel参数
            "nearest".to_string(),
            "sum".to_string(),
            fpn_params.2,
        )?;

        // construct Hiera
        let trunk = Hiera::new(
            vb.pp("image_encoder.trunk"),
            hiera_conf,
            image_size_pos_embed,
        )?;

        // construct ImageEncoder
        let image_encoder = ImageEncoder::new(trunk, neck, scalp)?;

        // construct MemoryAttention
        let memory_attention = MemoryAttention::new(
            ma_params.0,
            ma_params.1,
            ma_params.2,
            ma_params.3,
            true, // batch_first
            vb.pp("memory_attention"),
        )?;

        // 构建MemoryEncoder
        let memory_encoder = {
            // MaskDownSampler
            let mask_downsampler = MaskDownSampler::new(
                256,
                me_params.0 .0,
                me_params.0 .1,
                me_params.0 .2,
                16,
                candle_nn::Activation::Gelu,
                vb.pp("memory_encoder.mask_downsampler"),
            )?;

            // Fuser
            let fuser = Fuser::new(
                me_params.1 .1,
                me_params.1 .0, // 预读取的num_layers
                None,
                false,
                vb.pp("memory_encoder.fuser"),
            )?;

            // PositionEncoding
            let position_encoding = PositionEmbeddingSine::new(
                me_params.2 .0,
                me_params.2 .1,
                me_params.2 .2,
                None,
                true,
                image_size,
                &[4, 8, 16, 32],
                device,
            );

            MemoryEncoder::new(
                me_params.3,
                me_params.4,
                mask_downsampler,
                fuser,
                position_encoding,
                vb.pp("memory_encoder"),
            )?
        };

        Self::new(
            vb,
            image_encoder,
            memory_attention,
            memory_encoder,
            num_maskmem,
            image_size,
            use_high_res_features_in_sam,
            use_obj_ptrs_in_encoder,
            proj_tpos_enc_in_obj_ptrs, // TODO
            pred_obj_scores,
            directly_add_no_mem_embed,
            iou_prediction_use_sigmoid,
            use_mask_input_as_output_without_sam,
            max_cond_frames_in_attn,
            ext_config,
        )
    }

    pub fn get_device(&self) -> &Device {
        &self.device
    }

    pub fn get_image_size(&self) -> usize {
        self.image_size
    }

    pub fn device(&self) -> &Device {
        self.no_mem_embed.device()
    }

    pub fn _prepare_backbone_features(
        &self,
        backbone_fpn: &[Tensor],
        vision_pos_enc: &[Tensor],
    ) -> Result<(Vec<Tensor>, Vec<Tensor>, Vec<(usize, usize)>)> {
        if backbone_fpn.len() != vision_pos_enc.len() {
            return Err(candle_core::Error::Msg(
                "backbone_fpn and vision_pos_enc must have the same length".into(),
            ));
        }
        if backbone_fpn.len() < self.num_feature_levels {
            return Err(candle_core::Error::Msg(
                format!(
                    "backbone_fpn length {} < num_feature_levels {}",
                    backbone_fpn.len(),
                    self.num_feature_levels
                )
                .into(),
            ));
        }

        let start_idx = backbone_fpn.len() - self.num_feature_levels;
        let feature_maps = &backbone_fpn[start_idx..];
        let pos_embeddings = &vision_pos_enc[start_idx..];

        let feat_sizes = pos_embeddings
            .iter()
            .map(|x| {
                let dims = x.dims();
                if dims.len() < 2 {
                    Err(candle_core::Error::Msg(
                        "Tensor should have at least 2 dimensions".into(),
                    ))
                } else {
                    Ok((dims[dims.len() - 2], dims[dims.len() - 1]))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        let process_tensor = |t: &Tensor| -> Result<Tensor> {
            let dims = t.dims();
            let new_shape = (dims[0], dims[1], dims[2..].iter().product::<usize>());
            let t = t.reshape(new_shape)?;
            t.permute((2, 0, 1)) // [HW, N, C]
        };

        let vision_feats = feature_maps
            .iter()
            .map(process_tensor)
            .collect::<Result<Vec<_>>>()?;
        let vision_pos_embeds = pos_embeddings
            .iter()
            .map(process_tensor)
            .collect::<Result<Vec<_>>>()?;

        Ok((vision_feats, vision_pos_embeds, feat_sizes))
    }

    pub fn add_no_mem_embed(&self, vision_feat_last: &Tensor) -> Result<Tensor> {
        vision_feat_last.broadcast_add(&self.no_mem_embed)
    }

    pub fn forward_image(&self, img: &Tensor) -> Result<(Tensor, Vec<Tensor>, Vec<Tensor>)> {
        let mut backbone_out = self.image_encoder.forward(img)?;

        if self.use_high_res_features_in_sam {
            let (backbone_fpn_0, backbone_fpn_1) = self
                .sam_mask_decoder
                .forward_conv_s0_s1(&backbone_out.1[0], &backbone_out.1[1])?;
            backbone_out.1[0] = backbone_fpn_0;
            backbone_out.1[1] = backbone_fpn_1;
        }

        Ok(backbone_out)
    }

    pub fn prep_prompts(&self, prompts: &[Prompt], img_hw: (usize, usize)) -> (Tensor, Tensor) {
        let (h, w) = img_hw;

        let mut points = Vec::<(f32, f32, u8)>::new();
        let mut boxes = Vec::<(f32, f32, f32, f32)>::new();

        for prompt in prompts {
            match prompt {
                Prompt::Point(x, y, label) => {
                    points.push((*x as f32 / w as f32, *y as f32 / h as f32, label.clone()));
                }
                Prompt::Box(x1, y1, x2, y2) => {
                    boxes.push((
                        *x1 as f32 / w as f32,
                        *y1 as f32 / h as f32,
                        *x2 as f32 / w as f32,
                        *y2 as f32 / h as f32,
                    ));
                }
            }
        }

        self.prep_norm_prompts(&points, &boxes)
    }

    pub fn prep_norm_prompts(
        &self,
        points: &[(f32, f32, u8)],
        boxes: &[(f32, f32, f32, f32)], // 假设box格式为(x1, y1, x2, y2)
    ) -> (Tensor, Tensor) {
        let mut coords = Vec::<f32>::new();
        let mut labels = Vec::new();

        for &(x1, y1, x2, y2) in boxes {
            coords.extend(&[x1 * self.image_size as f32, y1 * self.image_size as f32]);
            labels.push(2f32);

            coords.extend(&[x2 * self.image_size as f32, y2 * self.image_size as f32]);
            labels.push(3f32);
        }

        for &(x, y, label) in points {
            coords.extend(&[x * self.image_size as f32, y * self.image_size as f32]);
            labels.push(label as f32);
        }

        let total_points = coords.len() / 2;
        let device = self.no_mem_embed.device();

        // Tensor [n, 2]
        let coord_tensor = Tensor::from_vec(coords, (1, total_points, 2), &device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();

        // Tensor [n]
        let label_tensor = Tensor::from_vec(labels, (1, total_points), &device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();

        (coord_tensor, label_tensor)
    }

    pub fn get_dense_pe(&self) -> Result<Tensor> {
        self.sam_prompt_encoder.get_dense_pe()
    }

    pub fn prompt_encoder_forward(
        &self,
        points: Option<&(Tensor, Tensor)>,
    ) -> Result<(Tensor, Tensor)> {
        let points = match points {
            Some(ps) => {
                if ps.0.dim(0)? == 0 {
                    None
                } else {
                    Some(ps)
                }
            }
            None => None,
        };
        self.sam_prompt_encoder.forward(points, None, None)
    }
    pub fn mask_decoder_forward(
        &self,
        image_embeddings: &Tensor,
        image_pe: &Tensor,
        sparse_prompt_embeddings: &Tensor,
        dense_prompt_embeddings: &Tensor,
        multimask_output: bool,
        high_res_features: Option<&(Tensor, Tensor)>,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        self.sam_mask_decoder.forward(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            multimask_output,
            high_res_features,
        )
    }

    pub fn _prepare_memory_conditioned_features(
        &self,
        frame_idx: usize,
        is_init_cond_frame: bool,
        current_vision_feats: &Tensor,
        current_vision_pos_embeds: &Tensor,
        feat_sizes: &[(usize, usize)],
        cond_frame_outputs: &BTreeMap<usize, FrameOutput>,
        non_cond_frame_outputs: &BTreeMap<usize, FrameOutput>,
        num_frames: usize,
        track_in_reverse: bool,
    ) -> Result<Tensor> {
        let device = self.device();
        let B = current_vision_feats.dim(1)?; // [H*W, B, C]
        let C = self.image_encoder.neck_d_model();
        let (H, W) = *feat_sizes.last().unwrap();

        // 处理无记忆情况（对齐Python的num_maskmem==0逻辑）
        if self.num_maskmem == 0 {
            return current_vision_feats
                .permute((1, 2, 0))?
                .reshape((B, C, H, W));
        }

        let mut debug: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();

        let mut to_cat_memory = vec![];
        let mut to_cat_memory_pos_embed = vec![];
        let mut num_obj_ptr_tokens = 0;

        if !is_init_cond_frame {
            let (selected_cond, unselected_cond) = select_closest_cond_frames(
                frame_idx,
                cond_frame_outputs,
                self.max_cond_frames_in_attn,
            );

            // 步骤2：构建时间位置和previous帧（修复prev_frame_idx计算）
            let mut t_pos_and_prevs = vec![];
            // 添加cond frames (t_pos=0)
            for (t, out) in &selected_cond {
                t_pos_and_prevs.push((0, *out));
            }

            // 添加non-cond frames (t_pos=1..num_maskmem-1)
            let stride = self.ext_config.memory_temporal_stride_for_eval;
            for t_pos in 1..self.num_maskmem {
                let t_rel = self.num_maskmem - t_pos;
                let prev_frame_idx = if t_rel == 1 {
                    // 处理t_rel == 1的特殊情况
                    if !track_in_reverse {
                        frame_idx - 1 // 正向：前一帧
                    } else {
                        frame_idx + 1 // 反向：后一帧
                    }
                } else {
                    if !track_in_reverse {
                        // 正向追踪，计算基址并减去偏移
                        let base = (frame_idx - 2) / stride * stride;
                        base - (t_rel - 2) * stride
                    } else {
                        // 反向追踪，计算基址并加上偏移
                        let base = ((frame_idx + 2) + stride - 1) / stride * stride;
                        base + (t_rel - 2) * stride
                    }
                };

                // 优先从non_cond中取，其次从未选中的cond中取
                let out = non_cond_frame_outputs
                    .get(&prev_frame_idx)
                    .or_else(|| unselected_cond.get(&prev_frame_idx).copied());

                if let Some(out) = out {
                    t_pos_and_prevs.push((t_pos, out));
                }
            }

            // 步骤3：收集memory特征和位置编码
            for (t_pos, prev) in t_pos_and_prevs {
                if let Some((feats, pos_enc)) = &prev.maskmem_features {
                    debug.insert(format!("feats_{}", t_pos), feats.clone());
                    debug.insert(format!("pos_enc_{}", t_pos), pos_enc.clone());
                    // 对齐PyTorch的flatten+permute操作
                    let mem_feat = feats
                        .to_dtype(DType::F32)?
                        .flatten(2, 3)?
                        .permute((2, 0, 1))?
                        .to_device(device)?; // [HW, B, C]
                    to_cat_memory.push(mem_feat);

                    // 处理位置编码（修复maskmem_tpos_enc索引）
                    let mut mem_pos = pos_enc
                        .flatten(2, 3)?
                        .permute((2, 0, 1))?
                        .to_device(device)?;
                    let enc_idx = self.num_maskmem - t_pos - 1;
                    mem_pos = mem_pos.broadcast_add(
                        //&self.maskmem_tpos_enc.narrow(0, enc_idx, 1)?,
                        &self.maskmem_tpos_enc.i(enc_idx)?,
                    )?;
                    to_cat_memory_pos_embed.push(mem_pos);
                }
            }

            // 步骤4：处理对象指针（修复时间编码和reshape逻辑）
            if self.ext_config.use_obj_ptrs_in_encoder {
                let max_ptrs = usize::min(num_frames, self.ext_config.max_obj_ptrs_in_encoder);

                // +++ 新增逻辑：过滤条件帧 +++
                let ptr_cond_outputs: Vec<_> = if self.ext_config.only_obj_ptrs_in_the_past_for_eval
                {
                    selected_cond
                        .iter()
                        .filter(|(&t, _)| {
                            if track_in_reverse {
                                t >= frame_idx // 反向追踪：只保留当前帧及之后
                            } else {
                                t <= frame_idx // 正向追踪：只保留当前帧及之前
                            }
                        })
                        .collect()
                } else {
                    selected_cond.iter().collect()
                };

                // 收集指针（修复track_in_reverse符号处理）
                let mut pos_and_ptrs = vec![];
                for (&t, out) in ptr_cond_outputs {
                    let t_diff = if self.ext_config.use_signed_tpos_enc_to_obj_ptrs {
                        if track_in_reverse {
                            (t as i32 - frame_idx as i32) as usize
                        } else {
                            (frame_idx as i32 - t as i32) as usize
                        }
                    } else {
                        frame_idx.abs_diff(t)
                    };
                    pos_and_ptrs.push((t_diff, &out.obj_ptr));
                }

                // 补充non-cond指针
                for t_diff in 1..max_ptrs {
                    let t = if track_in_reverse {
                        frame_idx + t_diff
                    } else {
                        frame_idx.saturating_sub(t_diff)
                    };
                    if let Some(out) = non_cond_frame_outputs.get(&t) {
                        pos_and_ptrs.push((t_diff, &out.obj_ptr));
                    }
                }

                if !pos_and_ptrs.is_empty() {
                    let (pos_list, ptrs_list): (Vec<_>, Vec<_>) = pos_and_ptrs
                        .into_iter()
                        .map(|(p, ptr)| (p, ptr.clone()))
                        .unzip();

                    // 生成时间编码（修复归一化逻辑）
                    let t_diff_max = max_ptrs.saturating_sub(1);
                    // TODO if self.add_tpos_enc_to_obj_ptrs:
                    let obj_pos = Tensor::new(
                        pos_list
                            .iter()
                            .map(|&v| v as f32 / t_diff_max as f32)
                            .collect::<Vec<_>>(),
                        device,
                    )?;
                    let tpos_dim = if self.ext_config.proj_tpos_enc_in_obj_ptrs {
                        C
                    } else {
                        self.mem_dim
                    };

                    let mut obj_pos_embed = get_1d_sine_pe(&obj_pos, tpos_dim, 10000.0)?;
                    obj_pos_embed = self.obj_ptr_tpos_proj.forward(&obj_pos_embed)?;
                    obj_pos_embed = obj_pos_embed.unsqueeze(1)?;

                    let obj_ptrs = Tensor::stack(&ptrs_list, 0)?;

                    // 处理mem_dim < C的情况（修复reshape顺序）
                    let (obj_ptrs, obj_pos_embed) = if self.mem_dim < C {
                        // 分割对象指针
                        let split_num = C / self.mem_dim;
                        let new_shape = (obj_ptrs.dim(0)?, B, split_num, self.mem_dim);
                        let obj_ptrs = obj_ptrs
                            .reshape(new_shape)?
                            .permute((0, 2, 1, 3))?
                            .flatten(0, 1)?;

                        // 扩展位置编码
                        let obj_pos_embed = obj_pos_embed
                            .repeat((split_num, 1, 1))?;

                        (obj_ptrs, obj_pos_embed)
                    } else {
                        (obj_ptrs, obj_pos_embed)
                    };

                    num_obj_ptr_tokens = obj_ptrs.dim(0)?;
                    to_cat_memory.push(obj_ptrs);
                    to_cat_memory_pos_embed.push(obj_pos_embed);
                }
            }
        } else if self.directly_add_no_mem_embed {
            // 初始帧特殊处理
            return current_vision_feats
                .broadcast_add(&self.no_mem_embed)?
                .permute((1, 2, 0))?
                .reshape((B, C, H, W));
        } else {
            // 添加无记忆标记
            to_cat_memory.push(self.no_mem_embed.broadcast_as((1, B, self.mem_dim))?);
            to_cat_memory_pos_embed.push(self.no_mem_pos_enc.broadcast_as((1, B, self.mem_dim))?);
        }

        let memory = Tensor::cat(&to_cat_memory, 0)?;
        let memory_pos = Tensor::cat(&to_cat_memory_pos_embed, 0)?;


        let fused = self.memory_attention.forward(
            current_vision_feats,
            Some(current_vision_pos_embeds),
            &memory,
            Some(&memory_pos),
            num_obj_ptr_tokens,
        )?;

        // reshape [H*W, B, C] -> [B, C, H, W]
        fused.permute((1, 2, 0))?.reshape((B, C, H, W))
    }

    pub fn track_step(
        &self,
        frame_idx: usize,
        is_init_cond_frame: bool,
        current_vision_feats: &[Tensor],
        current_vision_pos_embeds: &[Tensor],
        feat_sizes: &[(usize, usize)],
        point_inputs: Option<&(Tensor, Tensor)>,
        cond_frame_outputs: &BTreeMap<usize, FrameOutput>, // frame_idx -> frame_output
        non_cond_frame_outputs: &BTreeMap<usize, FrameOutput>, // frame_idx -> frame_output
        num_frames: usize,
        run_mem_encoder: bool,
        track_in_reverse: bool,
    ) -> Result<FrameOutput> {
        let feat = current_vision_feats.last().unwrap();
        let pos = current_vision_pos_embeds.last().unwrap();
        let pix_feat = self._prepare_memory_conditioned_features(
            frame_idx,
            is_init_cond_frame,
            feat,
            pos,
            feat_sizes,
            cond_frame_outputs,
            non_cond_frame_outputs,
            num_frames,
            track_in_reverse,
        )?;

        let high_res_features: Option<(Tensor, Tensor)> = if (current_vision_feats.len() > 2) {
            let feat1 = current_vision_feats[0].permute((1, 2, 0))?.reshape((
                1,
                current_vision_feats[0].dim(2)?, // 32
                feat_sizes[0].0,                 // 256
                feat_sizes[0].1,                 // 256
            ))?;

            let feat2 = current_vision_feats[1].permute((1, 2, 0))?.reshape((
                1,
                current_vision_feats[1].dim(2)?, // 64
                feat_sizes[1].0,                 // 128
                feat_sizes[1].1,                 // 128
            ))?;

            Some((feat1, feat2))
        } else {
            None
        };

        let num_ptr = match point_inputs {
            Some(ps) => ps.0.dim(1)?,
            None => 0usize,
        };
        let multimask_output = self._use_multimask(is_init_cond_frame, num_ptr);
        let sam_outputs = self.forward_sam_heads(
            &pix_feat,
            high_res_features.as_ref(),
            point_inputs,
            multimask_output,
        )?;

        let (_, _, _, low_res_masks, high_res_masks, obj_ptr, object_score_logits) = sam_outputs;

        let maskmem_features = self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            &high_res_masks,
            &object_score_logits,
            run_mem_encoder,
        )?;

        let out = FrameOutput {
            pred_masks: low_res_masks,
            pred_masks_high_res: high_res_masks,
            obj_ptr,
            object_score_logits,
            maskmem_features,
        };

        Ok(out)
    }

    pub fn forward_sam_heads(
        &self,
        backbone_features: &Tensor,
        high_res_features: Option<&(Tensor, Tensor)>,
        point_inputs: Option<&(Tensor, Tensor)>,
        multimask_output: bool,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        //) -> Result<()> {

        // TODO optimize
        let zero_points = (
            Tensor::zeros((1, 1, 2), DType::F32, backbone_features.device())?,
            (Tensor::ones((1, 1), DType::F32, backbone_features.device())? * -1.0)?,
        );
        let point_inputs = if point_inputs.is_none() {
            Some(&zero_points)
        } else {
            point_inputs
        };


        // 调用prompt encoder生成sparse和dense embeddings
        let (sparse_embeddings, dense_embeddings) = self.prompt_encoder_forward(point_inputs)?;

        // 获取dense positional encoding
        let image_pe = self.get_dense_pe()?;

        // 调用mask decoder得到输出
        let (low_res_multimasks_, ious, sam_output_tokens, object_score_logits) = self
            .mask_decoder_forward(
                backbone_features,
                &image_pe,
                &sparse_embeddings,
                &dense_embeddings,
                multimask_output,
                high_res_features,
            )?;


        // 3. 新增：处理NO_OBJ_SCORE逻辑
        let low_res_multimasks = if self.ext_config.pred_obj_scores {
            // 生成条件掩码 [B,]
            let is_obj_appearing = object_score_logits.gt(0.0)?;

            // 将条件扩展为 [B, 1, 1, 1] 以便广播
            //let mask = is_obj_appearing
            //    .unsqueeze(1)?
            //    .unsqueeze(2)?;
            let mask = is_obj_appearing.broadcast_as(low_res_multimasks_.shape())?;

            // 创建替换值张量
            let no_obj_tensor = Tensor::full(
                NO_OBJ_SCORE,
                low_res_multimasks_.shape(),
                low_res_multimasks_.device(),
            )?;

            // 执行条件替换
            mask.where_cond(&low_res_multimasks_, &no_obj_tensor)?
        } else {
            low_res_multimasks_
        };

        // 3. Interpolate to high resolution
        let high_res_multimasks =
            bilinear_interpolate_tensor(&low_res_multimasks, self.image_size, self.image_size)?;

        // 4. Select best masks based on IoU
        let (low_res_masks, high_res_masks) = if multimask_output {
            let b = low_res_multimasks.dim(0)?;
            //let best_iou_inds = ious.argmax(D::Minus1)?;
            //let batch_inds = Tensor::arange(0, b as u32, low_res_masks.device())?.to_dtype(DType::U32)?;

            let best_iou_inds = ious.argmax(1)?;
            let idx = best_iou_inds.unsqueeze(1)?.unsqueeze(2)?;

            let low_res_masks = low_res_multimasks
                .index_select(&best_iou_inds, 1)?;
            let high_res_masks = high_res_multimasks
                .index_select(&best_iou_inds, 1)?;

            (low_res_masks, high_res_masks)
        } else {
            (low_res_multimasks.clone(), high_res_multimasks.clone())
        };

        // 5. Process output tokens
        let sam_output_token = if multimask_output && sam_output_tokens.dim(1)? > 1 {
            //let best_iou_inds = ious.argmax(D::Minus1)?;
            let best_iou_inds = ious.argmax(1)?;
            let idx = best_iou_inds;
            sam_output_tokens
                //.index_select(&best_iou_inds.unsqueeze(1)?.unsqueeze(2)?, 1)?
                .index_select(&idx, 1)?
                .squeeze(1)?
        } else {
            sam_output_tokens.i((.., 0))?
        };

        // 6. Generate object pointer
        let mut obj_ptr = self.obj_ptr_proj.forward(&sam_output_token)?;

        // 7. Apply object score adjustments
        if self.ext_config.pred_obj_scores && self.no_obj_ptr.is_some() {
            let is_obj_appearing = object_score_logits.gt(0.0)?;

            let lambda = if self.ext_config.soft_no_obj_ptr {
                candle_nn::ops::sigmoid(&object_score_logits)?
            } else {
                is_obj_appearing.to_dtype(DType::F32)?
            };

            if self.ext_config.fixed_no_obj_ptr {
                obj_ptr = obj_ptr.broadcast_mul(&lambda)?;
            }

            let no_obj_component = (Tensor::ones_like(&lambda)? - &lambda)?
                .broadcast_mul(self.no_obj_ptr.as_ref().unwrap())?;
            obj_ptr = obj_ptr.add(&no_obj_component)?;
        }

        Ok((
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ))
    }

    pub fn _encode_memory_in_output(
        &self,
        current_vision_feats: &[Tensor],
        feat_sizes: &[(usize, usize)],
        high_res_masks: &Tensor,
        object_score_logits: &Tensor,

        run_mem_encoder: bool,
        //high_res_masks,
        //object_score_logits,
    ) -> Result<Option<(Tensor, Tensor)>> {
        if run_mem_encoder && self.num_maskmem > 0 {
            let mem_out = self._encode_new_memory(
                current_vision_feats,
                feat_sizes,
                high_res_masks,
                object_score_logits,
                true,
            )?;
            Ok(Some(mem_out))
        } else {
            Ok(None)
        }
    }

    pub fn _encode_new_memory(
        &self,
        current_vision_feats: &[Tensor],
        feat_sizes: &[(usize, usize)],
        high_res_masks: &Tensor,
        object_score_logits: &Tensor,
        is_mask_from_pts: bool,
    ) -> Result<(Tensor, Tensor)> {
        // 获取最后一个层级的视觉特征
        let last_feat = current_vision_feats
            .last()
            .ok_or(candle_core::Error::Msg(format!(
                "current_vision_feats is empty"
            )))?;
        let (s, b, c) = last_feat.dims3()?;
        let (h, w) = feat_sizes
            .last()
            .ok_or(candle_core::Error::Msg(format!("feat_sizes is empty")))?;
        if s != h * w {
            return Err(candle_core::Error::Msg(format!(
                "Feature size does not match H*W"
            )));
        }
        // 调整形状: (S, B, C) -> (B, C, H, W)
        let pix_feat = last_feat.permute((1, 2, 0))?.reshape((b, c, *h, *w))?;

        // 应用非重叠约束（如果需要）
        //let masks = if self.non_overlap_masks_for_mem_enc && !self.training {
        //    apply_non_overlapping_constraints(high_res_masks)?
        //} else {
        //    high_res_masks.clone()
        //};
        let masks = high_res_masks.clone();

        // 处理Sigmoid或二值化
        let binarize = self.ext_config.binarize_mask_from_pts_for_mem_enc && is_mask_from_pts;
        let mask_for_mem = if binarize {
            masks.gt(0.0)?.to_dtype(DType::F32)?
        } else {
            candle_nn::ops::sigmoid(&masks)?
        };

        // 应用比例和偏置
        let mask_for_mem = if self.ext_config.sigmoid_scale_for_mem_enc != 1.0 {
            mask_for_mem.broadcast_mul(&Tensor::new(
                self.ext_config.sigmoid_scale_for_mem_enc,
                last_feat.device(),
            )?)?
        } else {
            mask_for_mem
        };
        let mask_for_mem = if self.ext_config.sigmoid_bias_for_mem_enc != 0.0 {
            mask_for_mem.broadcast_add(&Tensor::new(
                self.ext_config.sigmoid_bias_for_mem_enc,
                last_feat.device(),
            )?)?
        } else {
            mask_for_mem
        };

        let mask_for_mem = mask_for_mem.broadcast_add(&Tensor::new(
            self.ext_config.sigmoid_bias_for_mem_enc,
            last_feat.device(),
        )?)?;

        // 调用记忆编码器
        let (mut maskmem_features, maskmem_pos_enc) =
            self.memory_encoder
                .forward(&pix_feat, &mask_for_mem, true)?;

        // 处理无对象嵌入
        if let Some(no_obj_embed) = &self.no_obj_embed_spatial {
            let is_obj_appearing = object_score_logits.gt(0.0)?;
            let is_obj_appearing = is_obj_appearing.to_dtype(DType::F32)?;
            let is_obj_appearing = is_obj_appearing.unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?; // B,1,1,1

            let one = Tensor::new(1.0, object_score_logits.device())?;
            let no_obj_factor = one.sub(&is_obj_appearing)?;

            // 扩展无对象嵌入到匹配的维度
            let no_obj_embed = no_obj_embed.unsqueeze(0)?.unsqueeze(2)?.unsqueeze(3)?; // 1,C,1,1
            let no_obj_embed = no_obj_embed.broadcast_as(maskmem_features.shape())?; // B,C,H,W

            maskmem_features = maskmem_features.add(&no_obj_factor.mul(&no_obj_embed)?)?;
        }

        Ok((maskmem_features, maskmem_pos_enc))
    }

    fn _use_multimask(&self, is_init_cond_frame: bool, num_pts: usize) -> bool {
        // Whether to use multimask output in the SAM head

        let multimask_output = (self.ext_config.multimask_output_in_sam
            && (is_init_cond_frame || self.ext_config.multimask_output_for_tracking)
            && (self.ext_config.multimask_min_pt_num <= num_pts)
            && (num_pts <= self.ext_config.multimask_max_pt_num));
        multimask_output
    }
}

fn select_closest_cond_frames<'a>(
    current_frame: usize,
    cond_frames: &'a BTreeMap<usize, FrameOutput>,
    max_count: usize,
) -> (
    BTreeMap<usize, &'a FrameOutput>,
    BTreeMap<usize, &'a FrameOutput>,
) {
    let mut selected = BTreeMap::new();
    let mut unselected = BTreeMap::new();

    // 处理无需选择的情况
    if max_count == usize::MAX || cond_frames.len() <= max_count {
        selected.extend(cond_frames.iter().map(|(t, out)| (*t, out)));
        return (selected, unselected);
    }

    // 利用BTreeMap的有序特性快速查找最近帧
    // 寻找最近前帧 (最大的小于current_frame的键)
    let closest_before = cond_frames
        .range(..current_frame)
        .next_back()
        .map(|(t, _)| *t);

    // 寻找最近后帧 (最小的大于等于current_frame的键)
    let closest_after = cond_frames.range(current_frame..).next().map(|(t, _)| *t);

    // 插入最近前后帧（直接使用引用）
    if let Some(t) = closest_before {
        selected.insert(t, cond_frames.get(&t).unwrap());
    }
    if let Some(t) = closest_after {
        selected.insert(t, cond_frames.get(&t).unwrap());
    }

    // 收集剩余候选帧（利用BTreeMap有序性优化）
    let remaining: Vec<_> = cond_frames
        .iter()
        .filter(|(&t, _)| {
            t != closest_before.unwrap_or(usize::MAX) && t != closest_after.unwrap_or(usize::MAX)
        })
        .collect();

    // 按时间差排序并选择
    let mut candidates: Vec<_> = remaining
        .into_iter()
        .map(|(t, out)| ((current_frame as isize - *t as isize).abs(), t, out))
        .collect();

    candidates.sort_by_key(|&(diff, _, _)| diff);

    for (_, t, out) in candidates.into_iter().take(max_count - selected.len()) {
        selected.insert(*t, out);
    }

    // 填充未选中的帧
    for (t, out) in cond_frames.iter() {
        if !selected.contains_key(t) {
            unselected.insert(*t, out);
        }
    }

    (selected, unselected)
}

pub fn apply_non_overlapping_constraints(pred_masks: &Tensor) -> Result<Tensor> {
    let batch_size = pred_masks.dims()[0];
    if batch_size == 1 {
        return Ok(pred_masks.clone());
    }

    let max_obj_inds = pred_masks.argmax(0)?;

    let batch_obj_inds = Tensor::arange(0f32, batch_size as f32, &pred_masks.device())?
        .reshape((batch_size, 1, 1, 1))?;

    let keep = max_obj_inds.eq(&batch_obj_inds)?;

    let suppressed_masks = pred_masks.clamp(-10.0, f32::INFINITY)?;

    let result = Tensor::where_cond(&keep, pred_masks, &suppressed_masks)?;

    Ok(result)
}
