use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder};

use crate::modeling::backbones::hieradet::Hiera;

// Define the ImageEncoder struct
pub struct ImageEncoder {
    trunk: Hiera,
    neck: FpnNeck,
    scalp: usize,
}

impl ImageEncoder {
    pub fn new(trunk: Hiera, neck: FpnNeck, scalp: usize) -> Result<Self> {
        // 验证通道数是否匹配
        if trunk.channel_list != neck.backbone_channel_list {
            return Err(candle_core::Error::Msg(format!(
                "Channel dims of trunk and neck do not match. Trunk: {:?}, Neck: {:?}",
                trunk.channel_list, neck.backbone_channel_list
            )));
        }

        Ok(Self { trunk, neck, scalp })
    }
    pub fn neck_d_model(&self) -> usize {
        self.neck.d_model()
    }

    pub fn forward(&self, sample: &Tensor) -> Result<(Tensor, Vec<Tensor>, Vec<Tensor>)> {
        // Forward through backbone
        let (features, pos) = self.neck.forward(&self.trunk.forward(sample)?)?;

        let (features, pos) = if self.scalp > 0 {
            // Discard the lowest resolution features
            (
                features[..features.len() - self.scalp].to_vec(),
                pos[..pos.len() - self.scalp].to_vec(),
            )
        } else {
            (features, pos)
        };
        let src = features.last().unwrap().clone();
        Ok((src, features, pos))
    }
}

// Define the FpnNeck struct
pub struct FpnNeck {
    position_encoding: Box<dyn Module>,
    convs: Vec<Conv2d>,
    pub backbone_channel_list: Vec<usize>,
    d_model: usize,
    fpn_interp_model: String,
    fuse_type: String,
    fpn_top_down_levels: Vec<usize>,
}

impl FpnNeck {
    pub fn new(
        vb: VarBuilder,
        position_encoding: Box<dyn Module>,
        d_model: usize,
        backbone_channel_list: Vec<usize>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        fpn_interp_model: String,
        fuse_type: String,
        fpn_top_down_levels: Option<Vec<usize>>,
    ) -> Result<Self> {
        let mut convs = Vec::new();
        for (idx, &dim) in backbone_channel_list.iter().enumerate() {
            let config = Conv2dConfig {
                stride,
                padding,
                ..Default::default()
            };
            let conv = candle_nn::conv2d(
                dim,
                d_model,
                kernel_size,
                config,
                vb.pp(format!("convs.{}.conv", idx)),
            )?;
            convs.push(conv);
        }
        let fpn_top_down_levels =
            fpn_top_down_levels.unwrap_or_else(|| (0..backbone_channel_list.len()).collect());
        Ok(Self {
            position_encoding,
            convs,
            backbone_channel_list,
            d_model,
            fpn_interp_model,
            fuse_type,
            fpn_top_down_levels,
        })
    }

    pub fn d_model(&self) -> usize {
        self.d_model
    }

    pub fn forward(&self, xs: &[Tensor]) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        let n = self.convs.len();
        let mut out = vec![None; n];
        let mut pos = vec![None; n];
        assert_eq!(xs.len(), n);


        let mut prev_features: Option<Tensor> = None;

        // Forward in top-down order (from low to high resolution)
        for i in (0..n).rev() {
            let x = &xs[i];
            let lateral_features = self.convs[n - i - 1].forward(x)?;

            let current_features =
                if self.fpn_top_down_levels.contains(&i) && prev_features.is_some() {
                    let prev_feat = prev_features.as_ref().unwrap();

                    let original_shape = prev_feat.dims4()?;
                    let h_target = original_shape.2 * 2;
                    let w_target = original_shape.3 * 2;
                    let top_down_features = prev_feat.upsample_nearest2d(h_target, w_target)?;

                    //let top_down_features = prev_features.as_ref().unwrap().upsample_nearest2d(target_h, target_w)?;
                    let fused = lateral_features.add(&top_down_features)?;
                    if self.fuse_type == "avg" {
                        (fused / 2.0)?
                    } else {
                        fused
                    }
                } else {
                    lateral_features
                };

            prev_features = Some(current_features.to_dtype(DType::F32)?);
            out[i] = Some(current_features.clone());
            pos[i] = Some(self.position_encoding.forward(&current_features)?);
        }

        let out : Vec<Tensor>= out.into_iter().map(|x| x.unwrap()).collect();
        let pos: Vec<Tensor> = pos.into_iter().map(|x| x.unwrap()).collect();

        Ok((out, pos))
    }

    pub fn backbone_channel_list(&self) -> Vec<usize> {
        self.backbone_channel_list.clone()
    }
}

