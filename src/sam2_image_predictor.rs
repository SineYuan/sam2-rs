use crate::modeling::interpolation::bilinear_interpolate_tensor;
use crate::modeling::sam2_base::*;
use crate::{preprocess_image, IMAGE_MEAN, IMAGE_STD};

use candle_core::{Result, Tensor};

use image::DynamicImage;

pub struct ImageEmbedding {
    original_img_size: (usize, usize),
    image_embed: Tensor,
    high_res_feats: (Tensor, Tensor),
}

pub struct SAM2ImagePredictor {
    model: SAM2Base,

    bb_feat_sizes: Vec<(usize, usize)>,
}

impl SAM2ImagePredictor {
    pub fn new(model: SAM2Base) -> Result<Self> {
        let bb_feat_sizes = vec![(256, 256), (128, 128), (64, 64)];
        Ok(Self {
            model: model,
            bb_feat_sizes,
        })
    }

    pub fn get_image_embedding(&self, image: &DynamicImage) -> Result<ImageEmbedding> {
        let origin_size = (image.height() as usize, image.width() as usize);
        let img_size = self.model.get_image_size();
        let img_tensor = preprocess_image(
            image,
            (img_size as u32, img_size as u32),
            &IMAGE_MEAN,
            &IMAGE_STD,
            self.model.get_device(),
        )?;

        self.get_preprocessed_image_tensor_embedding(&img_tensor, origin_size)
    }

    pub fn get_preprocessed_image_tensor_embedding(
        &self,
        image: &Tensor,
        orig_img_size: (usize, usize),
    ) -> Result<ImageEmbedding> {
        let backbone_out = self.model.forward_image(image)?;
        let (mut vision_feats, p2, p3) = self
            .model
            ._prepare_backbone_features(&backbone_out.1, &backbone_out.2)?;

        let last_idx = vision_feats.len() - 1;
        vision_feats[last_idx] = self.model.add_no_mem_embed(&vision_feats[last_idx])?;

        let mut feats: Vec<Tensor> = vision_feats
            .into_iter()
            .rev() // [::-1]
            .zip(self.bb_feat_sizes.iter().rev()) 
            .map(|(mut feat, &(h, w))| {
                // Permute (1, 2, 0)
                feat = feat.permute((1, 2, 0))?;

                // View: 1 × (C×H×W) × H_size × W_size
                // (C, H, W) -> (H, W, C)
                let (dim0, c, hw) = feat.shape().dims3()?;
                feat.reshape((1, c, h, w))
            })
            .collect::<Result<Vec<_>>>()?;

        // reverse  feats[::-1]
        feats.reverse();

        let image_embed = feats.last().unwrap().clone();
        let high_res_feats = feats[..feats.len() - 1].to_vec();
        if high_res_feats.len() != 2 {
            return Err(candle_core::Error::Msg(format!(
                "high_res_feats layer not 2"
            )));
        }

        let mut iter = high_res_feats.into_iter();
        let high_res_feats = (iter.next().unwrap(), iter.next().unwrap());

        let embed = ImageEmbedding {
            original_img_size: orig_img_size,
            image_embed: image_embed,
            high_res_feats: high_res_feats,
        };

        Ok(embed)
    }

    pub fn predict(
        &self,
        image_embed: &ImageEmbedding,
        prompts: &[Prompt],
        mask_threshold: f32,
        multimask_output: bool,
    ) -> Result<(Vec<Tensor>, Vec<f32>, Tensor)> {

        let original_img_size = &image_embed.original_img_size;
        let (img_embed, high_res_feats) = (&image_embed.image_embed, &image_embed.high_res_feats);

        let (points, labels) = self.model.prep_prompts(&prompts, *original_img_size);

        let (sparse_embeddings, dense_embeddings) =
            self.model.prompt_encoder_forward(Some(&(points, labels)))?;

        let image_pe = self.model.get_dense_pe()?;

        let (low_res_masks, iou_predictions, o2, o3) = self.model.mask_decoder_forward(
            img_embed,
            &image_pe,
            &sparse_embeddings,
            &dense_embeddings,
            multimask_output,
            Some(high_res_feats),
        )?;

        let masks =
            bilinear_interpolate_tensor(&low_res_masks, original_img_size.0, original_img_size.1)?;
        let masks = masks.gt(mask_threshold)?;
        let masks = masks.chunk(masks.dim(0)?, 0)?;
        // [1, 1, h, w] => [h, w]
        let masks = masks
            .into_iter()
            .map(|m| m.squeeze(0).unwrap().squeeze(0).unwrap())
            .collect();

        let ious = iou_predictions.squeeze(0)?.to_vec1::<f32>()?;
        Ok((masks, ious, low_res_masks))
    }
}
