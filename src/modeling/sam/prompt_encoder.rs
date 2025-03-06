use candle_core::{DType, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;

use crate::modeling::position_encoding::PositionEmbeddingRandom;
use crate::modeling::sam_utils::LayerNorm2d;

#[derive(Debug)]
pub struct PromptEncoder {
    pe_layer: PositionEmbeddingRandom,
    point_embeddings: Vec<candle_nn::Embedding>,
    not_a_point_embed: candle_nn::Embedding,
    mask_downscaling_conv1: candle_nn::Conv2d,
    mask_downscaling_ln1: LayerNorm2d,
    mask_downscaling_conv2: candle_nn::Conv2d,
    mask_downscaling_ln2: LayerNorm2d,
    mask_downscaling_conv3: candle_nn::Conv2d,
    no_mask_embed: candle_nn::Embedding,
    image_embedding_size: (usize, usize),
    input_image_size: (usize, usize),
    embed_dim: usize,
}

impl PromptEncoder {
    pub fn new(
        embed_dim: usize,
        image_embedding_size: (usize, usize),
        input_image_size: (usize, usize),
        mask_in_chans: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let num_points_embeddings = 4;
        let pe_layer = PositionEmbeddingRandom::new(embed_dim / 2, vb.pp("pe_layer"))?;
        let not_a_point_embed = candle_nn::embedding(1, embed_dim, vb.pp("not_a_point_embed"))?;
        let no_mask_embed = candle_nn::embedding(1, embed_dim, vb.pp("no_mask_embed"))?;
        let cfg = candle_nn::Conv2dConfig {
            stride: 2,
            ..Default::default()
        };
        let mask_downscaling_conv1 =
            candle_nn::conv2d(1, mask_in_chans / 4, 2, cfg, vb.pp("mask_downscaling.0"))?;
        let mask_downscaling_conv2 = candle_nn::conv2d(
            mask_in_chans / 4,
            mask_in_chans,
            2,
            cfg,
            vb.pp("mask_downscaling.3"),
        )?;
        let mask_downscaling_conv3 = candle_nn::conv2d(
            mask_in_chans,
            embed_dim,
            1,
            Default::default(),
            vb.pp("mask_downscaling.6"),
        )?;
        let mask_downscaling_ln1 =
            LayerNorm2d::new(mask_in_chans / 4, 1e-6, vb.pp("mask_downscaling.1"))?;
        let mask_downscaling_ln2 =
            LayerNorm2d::new(mask_in_chans, 1e-6, vb.pp("mask_downscaling.4"))?;
        let mut point_embeddings = Vec::with_capacity(num_points_embeddings);
        let vb_e = vb.pp("point_embeddings");
        for i in 0..num_points_embeddings {
            let emb = candle_nn::embedding(1, embed_dim, vb_e.pp(i))?;
            point_embeddings.push(emb)
        }
        Ok(Self {
            pe_layer,
            point_embeddings,
            not_a_point_embed,
            mask_downscaling_conv1,
            mask_downscaling_ln1,
            mask_downscaling_conv2,
            mask_downscaling_ln2,
            mask_downscaling_conv3,
            no_mask_embed,
            image_embedding_size,
            input_image_size,
            embed_dim,
        })
    }

    pub fn get_dense_pe(&self) -> Result<Tensor> {
        self.pe_layer
            .forward(self.image_embedding_size.0, self.image_embedding_size.1)?
            .unsqueeze(0)
    }

    fn embed_masks(&self, masks: &Tensor) -> Result<Tensor> {
        masks
            .apply(&self.mask_downscaling_conv1)?
            .apply(&self.mask_downscaling_ln1)?
            .gelu()?
            .apply(&self.mask_downscaling_conv2)?
            .apply(&self.mask_downscaling_ln2)?
            .gelu()?
            .apply(&self.mask_downscaling_conv3)
    }

    fn embed_points(&self, points: &Tensor, labels: &Tensor, pad: bool) -> Result<Tensor> {
        let points = (points + 0.5)?;
        let dev = points.device();
        let (points, labels) = if pad {
            let padding_point = Tensor::zeros((points.dim(0)?, 1, 2), DType::F32, dev)?;
            let padding_label = (Tensor::ones((labels.dim(0)?, 1), DType::F32, dev)? * (-1f64))?;
            let points = Tensor::cat(&[&points, &padding_point], 1)?;
            let labels = Tensor::cat(&[labels, &padding_label], 1)?;
            (points, labels)
        } else {
            (points, labels.clone())
        };
        let point_embedding = self
            .pe_layer
            .forward_with_coords(&points, self.input_image_size)?;
        let labels = labels.unsqueeze(2)?.broadcast_as(point_embedding.shape())?;
        let zeros = point_embedding.zeros_like()?;
        let point_embedding = labels.lt(0f32)?.where_cond(
            &self
                .not_a_point_embed
                .embeddings()
                .broadcast_as(zeros.shape())?,
            &point_embedding,
        )?;
        let labels0 = labels.eq(0f32)?.where_cond(
            &self.point_embeddings[0]
                .embeddings()
                .broadcast_as(zeros.shape())?,
            &zeros,
        )?;
        let point_embedding = (point_embedding + labels0)?;
        let labels1 = labels.eq(1f32)?.where_cond(
            &self.point_embeddings[1]
                .embeddings()
                .broadcast_as(zeros.shape())?,
            &zeros,
        )?;
        let point_embedding = (point_embedding + labels1)?;
        let labels2 = labels.eq(2f32)?.where_cond(
            &self.point_embeddings[2]
                .embeddings()
                .broadcast_as(zeros.shape())?,
            &zeros,
        )?;
        let point_embedding = (point_embedding + labels2)?;
        let labels3 = labels.eq(3f32)?.where_cond(
            &self.point_embeddings[3]
                .embeddings()
                .broadcast_as(zeros.shape())?,
            &zeros,
        )?;
        let point_embedding = (point_embedding + labels3)?;
        Ok(point_embedding)
    }

    fn embed_boxes(&self, boxes: &Tensor) -> Result<Tensor> {
        let boxes = (boxes + 0.5)?;
        let coords = boxes.reshape(((), 2, 2))?;
        let corner_embedding = self
            .pe_layer
            .forward_with_coords(&coords, self.input_image_size)?;
        let ce1 = corner_embedding.i((.., 0))?;
        let ce2 = corner_embedding.i((.., 1))?;
        let ce1 = (ce1 + self.point_embeddings[2].embeddings())?;
        let ce2 = (ce2 + self.point_embeddings[3].embeddings())?;
        Tensor::cat(&[&ce1, &ce2], 1)
    }

    fn get_batch_size(
        &self,
        points: Option<&(Tensor, Tensor)>,
        boxes: Option<&Tensor>,
        masks: Option<&Tensor>,
    ) -> Result<usize> {
        if let Some((coords, _)) = points {
            Ok(coords.dim(0)?)
        } else if let Some(boxes) = boxes {
            Ok(boxes.dim(0)?)
        } else if let Some(masks) = masks {
            Ok(masks.dim(0)?)
        } else {
            Ok(1)
        }
    }

    pub fn forward(
        &self,
        points: Option<&(Tensor, Tensor)>,
        boxes: Option<&Tensor>,
        masks: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let bs = self.get_batch_size(points, boxes, masks)?;

        let se_points = match points {
            Some((coords, labels)) => Some(self.embed_points(coords, labels, boxes.is_none())?),
            None => None,
        };
        let se_boxes = match boxes {
            Some(boxes) => Some(self.embed_boxes(boxes)?),
            None => None,
        };

        let sparse_embeddings = match (se_points, se_boxes) {
            (Some(se_points), Some(se_boxes)) => Tensor::cat(&[se_points, se_boxes], 1)?,
            (Some(se_points), None) => se_points,
            (None, Some(se_boxes)) => se_boxes,
            (None, None) => {
                let dev = self.no_mask_embed.embeddings().device();
                Tensor::zeros((bs, 0, self.embed_dim), DType::F32, dev)?
            }
        };

        let dense_embeddings = match masks {
            None => {
                let emb = self.no_mask_embed.embeddings();
                emb.reshape((1, self.embed_dim, 1, 1))?.expand((
                    bs,
                    self.embed_dim,
                    self.image_embedding_size.0,
                    self.image_embedding_size.1,
                ))?
            }
            Some(masks) => self.embed_masks(masks)?,
        };

        Ok((sparse_embeddings, dense_embeddings))
    }
}
