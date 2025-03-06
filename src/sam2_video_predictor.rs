use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use crate::{preprocess_image, IMAGE_MEAN, IMAGE_STD};
use candle_core::{DType, Device, Result, Tensor, D};

use image::{DynamicImage, GenericImageView};

use crate::modeling::interpolation::*;
use crate::modeling::sam2_base::*;

#[derive(Debug)]
pub struct ObjectState {
    cond_frame_outputs: BTreeMap<usize, FrameOutput>,
    non_cond_frame_outputs: BTreeMap<usize, FrameOutput>,

    temp_cond_frame_outputs: BTreeMap<usize, FrameOutput>,
    temp_non_cond_frame_outputs: BTreeMap<usize, FrameOutput>,

    point_inputs: BTreeMap<usize, (Tensor, Tensor)>,

    frames_tracked: BTreeMap<usize, bool>,
}

impl Default for ObjectState {
    fn default() -> Self {
        Self {
            cond_frame_outputs: BTreeMap::new(),
            non_cond_frame_outputs: BTreeMap::new(),
            temp_cond_frame_outputs: BTreeMap::new(),
            temp_non_cond_frame_outputs: BTreeMap::new(),
            point_inputs: BTreeMap::new(),
            frames_tracked: BTreeMap::new(),
        }
    }
}

// 状态管理结构
pub struct InferenceState {
    frame_loader: Box<dyn FrameLoader>,

    num_frames: usize,
    video_height: usize,
    video_width: usize,

    objects: BTreeMap<usize, ObjectState>,

    device: Device,
    storage_device: Device,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,

    cached_features: BTreeMap<usize, BackboneOutput>,

    obj_id_counter: usize,
    obj_id_to_idx: BTreeMap<usize, usize>,
    obj_idx_to_id: BTreeMap<usize, usize>,
}

impl InferenceState {
    pub fn reset(&mut self) {
        self.obj_id_counter = 0;
        self.objects.clear();
        self.obj_id_to_idx.clear();
        self.obj_idx_to_id.clear();
    }

    pub fn get_frame(&self, frame_idx: usize, target_size: (usize, usize)) -> Result<Tensor> {
        self.frame_loader
            .get_frame(frame_idx, target_size, &self.device)
    }

    pub fn save_features<P: AsRef<std::path::Path>>(&self, checkpoint_path: P) -> Result<()> {
        let mut features = HashMap::new();
        for (idx, bbout) in self.cached_features.iter() {
            for (i, tensor) in bbout.backbone_fpn.iter().enumerate() {
                features.insert(format!("{}.backbone_fpn.{}", idx, i), tensor.clone());
            }
            for (i, tensor) in bbout.vision_pos_enc.iter().enumerate() {
                features.insert(format!("{}.vision_pos_enc.{}", idx, i), tensor.clone());
            }
        }

        candle_core::safetensors::save(&features, checkpoint_path)
    }

    pub fn load_features<P: AsRef<std::path::Path>>(&mut self, checkpoint_path: P) -> Result<()> {
        let data = candle_core::safetensors::load(checkpoint_path, &self.storage_device)?;

        let mut temp_map: BTreeMap<usize, (Vec<Tensor>, Vec<Tensor>)> = BTreeMap::new();

        for (key, tensor) in data.into_iter() {
            let parts: Vec<&str> = key.split('.').collect();
            if parts.len() != 3 {
                return Err(candle_core::Error::Msg(format!(
                    "Invalid key format: {}",
                    key
                )));
            }
            let idx_str = parts[0];
            let tensor_type = parts[1];
            let i_str = parts[2];

            let idx = idx_str.parse::<usize>().map_err(|e| {
                candle_core::Error::Msg(format!("Failed to parse index from key {}: {}", key, e))
            })?;
            let i = i_str.parse::<usize>().map_err(|e| {
                candle_core::Error::Msg(format!("Failed to parse i from key {}: {}", key, e))
            })?;

            let entry = temp_map.entry(idx).or_insert((Vec::new(), Vec::new()));

            match tensor_type {
                "backbone_fpn" => {
                    if entry.0.len() != i {
                        return Err(candle_core::Error::Msg(format!(
                            "For idx {} backbone_fpn, expected index {} but got {}",
                            idx,
                            entry.0.len(),
                            i
                        )));
                    }
                    entry.0.push(tensor);
                }
                "vision_pos_enc" => {
                    if entry.1.len() != i {
                        return Err(candle_core::Error::Msg(format!(
                            "For idx {} vision_pos_enc, expected index {} but got {}",
                            idx,
                            entry.1.len(),
                            i
                        )));
                    }
                    entry.1.push(tensor);
                }
                _ => {
                    return Err(candle_core::Error::Msg(format!(
                        "Invalid tensor type '{}' in key {}",
                        tensor_type, key
                    )))
                }
            }
        }

        self.cached_features.clear(); // 清空原有数据或合并策略按需调整
        for (idx, (backbone_fpn, vision_pos_enc)) in temp_map {
            let bb_output = BackboneOutput {
                backbone_fpn,
                vision_pos_enc,
            };
            self.cached_features.insert(idx, bb_output);
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
struct BackboneOutput {
    backbone_fpn: Vec<Tensor>,
    vision_pos_enc: Vec<Tensor>,
}

pub struct ObjectMask {
    pub obj_id: usize,
    pub mask: Tensor,
}

pub struct SAM2VideoPredictor {
    base: SAM2Base,
    non_overlap_masks: bool,
    clear_non_cond_mem_around_input: bool,
    add_all_frames_to_correct_as_cond: bool,
}

impl SAM2VideoPredictor {
    pub fn new(
        base: SAM2Base,
        non_overlap_masks: bool,
        clear_non_cond_mem_around_input: bool,
        add_all_frames_to_correct_as_cond: bool,
    ) -> Self {
        Self {
            base,
            non_overlap_masks,
            clear_non_cond_mem_around_input,
            add_all_frames_to_correct_as_cond,
        }
    }

    pub fn init_state(
        &self,
        frame_loader: Box<dyn FrameLoader>,
        offload_video_to_cpu: bool,
        offload_state_to_cpu: bool,
    ) -> Result<InferenceState> {
        let compute_device = self.base.device();
        let video_device = if offload_video_to_cpu {
            &Device::Cpu
        } else {
            compute_device
        };

        // 加载视频帧
        //let (images, video_height, video_width) = load_video_frames(video_path, video_device)?;
        let (video_height, video_width) = frame_loader.frame_size();

        let num_frames = frame_loader.total_frames();
        let mut state = InferenceState {
            frame_loader,
            objects: BTreeMap::new(),
            num_frames: num_frames,
            video_height,
            video_width,
            device: compute_device.clone(),
            storage_device: if offload_state_to_cpu {
                Device::Cpu.clone()
            } else {
                compute_device.clone()
            },
            offload_video_to_cpu,
            offload_state_to_cpu,
            cached_features: BTreeMap::new(),
            obj_id_to_idx: BTreeMap::new(),
            obj_idx_to_id: BTreeMap::new(),
            obj_id_counter: 0,
        };

        // 预热第一帧的特征
        let img_size = self.base.get_image_size();
        let img_tensor = state.get_frame(0, (img_size, img_size))?;
        self.get_image_feature(Some(&img_tensor), &mut state.cached_features, 0)?;
        Ok(state)
    }

    // 实现对象ID映射方法
    fn obj_id_to_idx(&self, state: &mut InferenceState, obj_id: usize) -> usize {
        if let Some(&idx) = state.obj_id_to_idx.get(&obj_id) {
            return idx;
        }

        // 分配新索引（保持插入顺序）
        let new_idx = state.obj_id_to_idx.len();
        state.obj_id_to_idx.insert(obj_id, new_idx);
        state.obj_idx_to_id.insert(new_idx, obj_id);

        state.objects.insert(new_idx, ObjectState::default());
        new_idx
    }

    fn get_image_feature(
        &self,
        image: Option<&Tensor>,
        cached_features: &mut BTreeMap<usize, BackboneOutput>,
        frame_idx: usize,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>, Vec<(usize, usize)>)> {
        match cached_features.get(&frame_idx) {
            None => {
                if image.is_none() {
                    return Err(candle_core::Error::Msg(format!(
                        "No image input or cached feature for idx: {}",
                        frame_idx
                    )));
                }
                let (img_emb, backbone_fpn, vision_pos_enc) =
                    self.base.forward_image(image.unwrap())?;
                let bbout = BackboneOutput {
                    backbone_fpn,
                    vision_pos_enc,
                };
                let out = self
                    .base
                    ._prepare_backbone_features(&bbout.backbone_fpn, &bbout.vision_pos_enc)?;
                cached_features.insert(frame_idx, bbout);

                Ok(out)
            }
            Some(bbout) => self
                .base
                ._prepare_backbone_features(&bbout.backbone_fpn, &bbout.vision_pos_enc),
        }
    }

    pub fn add_new_points_or_box(
        &self,
        state: &mut InferenceState,
        frame_idx: usize,
        obj_id: usize,
        prompts: &[Prompt],
        clear_old_points: bool,
    ) -> Result<Vec<ObjectMask>> {
        // 参数校验
        let has_box = prompts.iter().any(|p| matches!(p, Prompt::Box(_, _, _, _)));
        if has_box && !clear_old_points {
            return Err(candle_core::Error::Msg(
                "Cannot add box without clearing old points".into(),
            ));
        }

        let obj_idx = self.obj_id_to_idx(state, obj_id);

        // 归一化坐标
        let video_w = state.video_width;
        let video_h = state.video_height;

        let (points, labels) = self.base.prep_prompts(&prompts, (video_h, video_w));

        let (points, labels) = {
            let mut obj_state = state.objects.get_mut(&obj_idx).unwrap();

            let entry = if clear_old_points {
                // clean old points
                obj_state.point_inputs.remove(&frame_idx);
                obj_state.point_inputs.entry(frame_idx)
            } else {
                obj_state.point_inputs.entry(frame_idx)
            };

            entry
                .and_modify(|(existing_pts, existing_lbls)| {
                    // merge point inputs
                    if points.dim(0).unwrap_or(0) > 0 || labels.dim(0).unwrap_or(0) > 0 {
                        *existing_pts = Tensor::cat(&[&(*existing_pts), &points], 1).unwrap();
                        *existing_lbls = Tensor::cat(&[&(*existing_lbls), &labels], 1).unwrap();
                    }
                })
                .or_insert_with(|| {
                    if clear_old_points
                        || (points.dim(0).unwrap_or(0) == 0 && labels.dim(0).unwrap_or(0) == 0)
                    {
                        // empty points tensor
                        (
                            Tensor::zeros((0, 0, 2), DType::F32, &state.device).unwrap(),
                            Tensor::zeros((0, 0), DType::F32, &state.device).unwrap(),
                        )
                    } else {
                        (points.clone(), labels.clone())
                    }
                });

            let (pts, lbls) = obj_state.point_inputs.get(&frame_idx).unwrap();
            (pts.clone(), lbls.clone())
        };

        // is_init_cond_frame = frame_idx not in obj_frames_tracked
        let is_init_cond_frame = true; // TODO

        let img_features = {
            let img_size = self.base.get_image_size();
            let img_tensor = state.get_frame(frame_idx, (img_size, img_size))?;
            self.get_image_feature(Some(&img_tensor), &mut state.cached_features, frame_idx)?
        };

        let obj_state = state.objects.get(&obj_idx).unwrap();

        // 运行单帧推理
        let current_out = {
            self.run_single_frame_inference(
                img_features,
                obj_state,
                frame_idx,
                obj_idx,
                Some(&(points, labels)),
                is_init_cond_frame,
                false,
                false,
            )?
        };

        let mut obj_state = state.objects.get_mut(&obj_idx).unwrap();
        let is_cond = is_init_cond_frame || self.add_all_frames_to_correct_as_cond;
        if is_cond {
            obj_state
                .temp_cond_frame_outputs
                .insert(frame_idx, current_out);
        } else {
            obj_state
                .temp_non_cond_frame_outputs
                .insert(frame_idx, current_out);
        };

        //storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        let out = self._consolidate_temp_output_across_obj(&state, frame_idx, is_cond, true)?;

        let (_, video_res_masks) = self._get_orig_video_res_output(&state, &out)?;

        let mut obj_masks = Vec::new();
        let masks = video_res_masks.chunk(video_res_masks.dim(0)?, 0)?;
        for (obj_idx, mask) in (0..state.obj_id_to_idx.len())
            .into_iter()
            .zip(masks.into_iter())
        {
            let obj_id = state.obj_idx_to_id.get(&obj_idx).unwrap();

            obj_masks.push(ObjectMask {
                obj_id: obj_id.clone(),
                mask: mask.squeeze(0)?.squeeze(0)?,
            });
        }

        Ok(obj_masks)
    }

    // 调整后的推理方法
    fn run_single_frame_inference(
        &self,
        img_features: (Vec<Tensor>, Vec<Tensor>, Vec<(usize, usize)>),
        obj_state: &ObjectState,
        frame_idx: usize,
        num_frames: usize,
        //obj_idx: usize,
        point_inputs: Option<&(Tensor, Tensor)>,
        is_init_cond_frame: bool,
        run_mem_encoder: bool,
        reverse: bool,
    ) -> Result<FrameOutput> {
        let (current_vision_feats, current_vision_pos_embeds, feat_sizes) = img_features;

        let mut current_out = self.base.track_step(
            frame_idx,
            is_init_cond_frame,
            &current_vision_feats,
            &current_vision_pos_embeds,
            &feat_sizes,
            point_inputs,
            &obj_state.cond_frame_outputs,
            &obj_state.non_cond_frame_outputs,
            num_frames,
            run_mem_encoder,
            reverse,
        )?;

        //let storage_device = &state.storage_device;
        // TODO offload to storage_device

        Ok(current_out)
    }

    pub fn _consolidate_temp_output_across_obj(
        &self,
        inference_state: &InferenceState,
        frame_idx: usize,
        is_cond: bool,
        consolidate_at_video_res: bool,
    ) -> Result<Tensor> {
        let batch_size = inference_state.obj_idx_to_id.len();
        let storage_key = if is_cond { "cond" } else { "non_cond" };

        // 确定输出分辨率
        let (h, w) = if consolidate_at_video_res {
            (inference_state.video_height, inference_state.video_width)
        } else {
            let size = self.base.get_image_size() / 4;
            (size, size)
        };

        // 初始化合并后的mask张量
        let mut consolidated = Tensor::full(
            NO_OBJ_SCORE,
            (batch_size, 1, h, w),
            &inference_state.storage_device,
        )?;

        for obj_idx in 0..batch_size {
            let obj_state = inference_state.objects.get(&obj_idx).unwrap();
            let mask = {
                (if storage_key == "cond" {
                    obj_state.temp_cond_frame_outputs.get(&frame_idx)
                } else {
                    obj_state.temp_non_cond_frame_outputs.get(&frame_idx)
                })
                .or_else(|| {
                    if storage_key == "cond" {
                        obj_state.cond_frame_outputs.get(&frame_idx)
                    } else {
                        obj_state.non_cond_frame_outputs.get(&frame_idx)
                    }
                })
            };

            if let Some(frame_out) = mask {
                let mut mask_tensor = frame_out.pred_masks.clone();

                // 调整分辨率
                if mask_tensor.dim(D::Minus2)? != h || mask_tensor.dim(D::Minus1)? != w {
                    mask_tensor = bilinear_interpolate_tensor(&mask_tensor, h, w)?;
                }

                // 设备对齐
                mask_tensor = mask_tensor.to_device(&inference_state.storage_device)?;

                // 更新合并后的张量
                consolidated = consolidated.slice_assign(
                    &[
                        obj_idx..obj_idx + 1, // 第0维：当前对象
                        0..1,                 // 第1维：单通道
                        0..h,                 // 第2维：全高度
                        0..w,                 // 第3维：全宽度
                    ],
                    &mask_tensor.reshape((1, 1, h, w))?,
                )?;
            }
        }

        Ok(consolidated)
    }

    pub fn _get_orig_video_res_output(
        &self,
        inference_state: &InferenceState,
        any_res_masks: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let video_h = inference_state.video_height;
        let video_w = inference_state.video_width;

        // Move to target device and check resolution
        let any_res_masks = any_res_masks.to_device(&inference_state.device)?;
        let video_res_masks =
            if any_res_masks.dim(2)? == video_h && any_res_masks.dim(3)? == video_w {
                any_res_masks.clone()
            } else {
                bilinear_interpolate_tensor(&any_res_masks, video_h, video_w)?
            };

        // Apply non-overlap constraints if enabled
        let final_masks = if self.non_overlap_masks {
            apply_non_overlapping_constraints(&video_res_masks)?
        } else {
            video_res_masks
        };

        Ok((any_res_masks, final_masks))
    }

    pub fn propagate_in_video_preflight(&self, state: &mut InferenceState) -> Result<()> {
        // 检查是否有对象
        let batch_size = state.obj_idx_to_id.len();
        if batch_size == 0 {
            return Err(candle_core::Error::Msg(format!(
                "No input points or masks provided for any object"
            )));
        }

        // 遍历所有对象
        for obj_idx in 0..batch_size {
            // 获取临时输出和主输出

            let obj_state = state.objects.get_mut(&obj_idx).ok_or_else(|| {
                candle_core::Error::Msg(format!("object index {} not found!", obj_idx))
            })?;

            // 处理条件帧和非条件帧
            for is_cond in &[true, false] {
                let storage_key = if *is_cond {
                    &mut obj_state.temp_cond_frame_outputs
                } else {
                    &mut obj_state.temp_non_cond_frame_outputs
                };

                // 遍历临时帧输出
                let frame_indices: Vec<usize> = storage_key.keys().cloned().collect();
                for frame_idx in frame_indices {
                    let mut out: FrameOutput = storage_key.remove(&frame_idx).unwrap();

                    // 生成内存特征
                    if out.maskmem_features.is_none() {
                        // 插值操作

                        let high_res_masks = bilinear_interpolate_tensor(
                            &out.pred_masks.to_device(&state.device)?,
                            self.base.get_image_size(),
                            self.base.get_image_size(),
                        )?;

                        // 调用内存编码核心逻辑
                        let (vision_feats, pos_enc, feat_sizes) = {
                            let img_size = self.base.get_image_size();
                            let img_tensor = state.frame_loader.get_frame(
                                frame_idx,
                                (img_size, img_size),
                                &state.device,
                            )?;
                            self.get_image_feature(
                                Some(&img_tensor),
                                &mut state.cached_features,
                                frame_idx,
                            )?
                        };
                        let (maskmem_features, maskmem_pos_enc) = self.base._encode_new_memory(
                            &vision_feats,
                            &feat_sizes,
                            &high_res_masks,
                            &out.object_score_logits,
                            true,
                        )?;

                        // 转换存储类型
                        out.maskmem_features = Some((
                            maskmem_features
                                .to_dtype(DType::BF16)?
                                .to_device(&state.storage_device)?,
                            //maskmem_features.to_device(&state.storage_device)?,
                            maskmem_pos_enc,
                        ));
                    }

                    // 存入主输出
                    let target_map = if *is_cond {
                        &mut obj_state.cond_frame_outputs
                    } else {
                        &mut obj_state.non_cond_frame_outputs
                    };
                    target_map.insert(frame_idx, out);

                    // 清理周围非条件内存
                    //if self.clear_non_cond_mem_around_input {
                    //    self.clear_non_cond_mem_around(state, frame_idx, obj_idx)?;
                    //}
                }
            }

            // 最终检查条件帧
            if obj_state.cond_frame_outputs.is_empty() {
                let obj_id = state.obj_idx_to_id.get(&obj_idx).unwrap();
                return Err(candle_core::Error::Msg(format!(
                    "No inputs for object {}",
                    obj_id
                )));
            }

            // 清理重复的非条件帧
            for frame_idx in obj_state.cond_frame_outputs.keys() {
                obj_state.non_cond_frame_outputs.remove(frame_idx);
            }
        }

        Ok(())
    }

    pub fn propagate_in_video(
        &self,
        inference_state: &mut InferenceState,
        start_frame_idx: Option<usize>,
        max_frame_num_to_track: Option<usize>,
        reverse: bool,
    ) -> Result<Vec<(usize, Vec<ObjectMask>)>> {
        self.propagate_in_video_preflight(inference_state)?;

        let num_frames = inference_state.num_frames;
        let batch_size = inference_state.obj_idx_to_id.len();
        let mut results = Vec::new();

        // 确定起始帧和结束帧
        let start_frame_idx = start_frame_idx.unwrap_or_else(|| {
            inference_state
                .objects
                .values()
                .flat_map(|obj| {
                    obj.cond_frame_outputs
                        .keys()
                        .chain(obj.temp_cond_frame_outputs.keys())
                })
                .min()
                .cloned()
                .unwrap_or(0)
        });

        let max_frame_num = max_frame_num_to_track.unwrap_or(num_frames);
        let (end_frame_idx, processing_order) = if reverse {
            let end = start_frame_idx.saturating_sub(max_frame_num);
            (end, (end..=start_frame_idx).rev().collect::<Vec<_>>())
        } else {
            let end = (start_frame_idx + max_frame_num).min(num_frames - 1);
            (end, (start_frame_idx..=end).collect::<Vec<_>>())
        };

        for &frame_idx in &processing_order {
            let mut pred_masks_per_obj = Vec::with_capacity(batch_size);

            for obj_idx in 0..batch_size {
                let obj_state = inference_state
                    .objects
                    .get_mut(&obj_idx)
                    .ok_or_else(|| candle_core::Error::Msg("Object state not found".to_string()))?;

                // 检查是否已经存在条件帧输出
                if let Some(current_out) = obj_state
                    .cond_frame_outputs
                    .get(&frame_idx)
                    .or_else(|| obj_state.temp_cond_frame_outputs.get(&frame_idx))
                {
                    let pred_masks = current_out.pred_masks.to_device(&inference_state.device)?;
                    pred_masks_per_obj.push(pred_masks);

                    if self.clear_non_cond_mem_around_input {
                        self.clear_non_cond_mem_around(inference_state, frame_idx, obj_idx)?;
                    }
                } else {
                    let img_features = {
                        let img_size = self.base.get_image_size();
                        let img_tensor = inference_state.frame_loader.get_frame(
                            frame_idx,
                            (img_size, img_size),
                            &inference_state.device,
                        )?;
                        self.get_image_feature(
                            Some(&img_tensor),
                            &mut inference_state.cached_features,
                            frame_idx,
                        )?
                    };
                    // 运行单帧推理
                    let current_out = self.run_single_frame_inference(
                        img_features,
                        &obj_state,
                        frame_idx,
                        inference_state.num_frames,
                        None,
                        false,
                        true,
                        reverse,
                    )?;

                    let pred_masks = current_out.pred_masks.to_device(&inference_state.device)?;
                    pred_masks_per_obj.push(pred_masks);

                    // 存储到非条件帧输出
                    obj_state
                        .non_cond_frame_outputs
                        .insert(frame_idx, current_out);
                }

                inference_state
                    .objects
                    .get_mut(&obj_idx)
                    .unwrap()
                    .frames_tracked
                    .insert(frame_idx, reverse);
            }

            // 合并所有对象的预测掩码
            let all_pred_masks = Tensor::cat(&pred_masks_per_obj, 0)?;
            let (_, video_res_masks) =
                self._get_orig_video_res_output(inference_state, &all_pred_masks)?;

            // 收集结果：帧索引、对象ID列表、掩码张量
            let obj_ids = inference_state
                .obj_idx_to_id
                .values()
                .cloned()
                .collect::<Vec<_>>();
            let masks = video_res_masks.chunk(obj_ids.len(), 0)?;
            let mut obj_masks = Vec::new();
            for (obj_id, mask) in obj_ids.into_iter().zip(masks.into_iter()) {
                obj_masks.push(ObjectMask {
                    obj_id: obj_id.clone(),
                    mask: mask.squeeze(0)?.squeeze(0)?,
                });
            }

            results.push((frame_idx, obj_masks));
        }

        Ok(results)
    }

    fn clear_non_cond_mem_around(
        &self,
        state: &mut InferenceState,
        frame_idx: usize,
        obj_idx: usize,
    ) -> Result<()> {
        let clear_range = frame_idx.saturating_sub(2)..=frame_idx.saturating_add(2);
        if let Some(obj_state) = state.objects.get_mut(&obj_idx) {
            obj_state
                .non_cond_frame_outputs
                .retain(|k, _| !clear_range.contains(k));
        }
        Ok(())
    }
}

// 定义 trait
pub trait FrameLoader {
    fn get_frame(
        &self,
        index: usize,
        target_size: (usize, usize),
        device: &Device,
    ) -> Result<Tensor>;
    fn frame_size(&self) -> (usize, usize);
    fn total_frames(&self) -> usize;
}

pub struct ImageLoader {
    image_paths: Vec<PathBuf>,
    base_dimensions: (u32, u32),
}

impl ImageLoader {
    pub fn new<P: AsRef<Path>>(folder_path: P) -> Result<Self> {
        let mut image_paths = Vec::new();
        let supported_extensions = ["jpg", "jpeg", "png", "gif", "bmp", "webp"];

        // 读取目录并收集图片路径
        let entries = std::fs::read_dir(folder_path)
            .map_err(|e| candle_core::Error::wrap(format!("read dir fail: {e}")))?;

        for entry in entries {
            let entry = entry.map_err(|e| candle_core::Error::wrap(format!("item error: {e}")))?;
            let path = entry.path();

            if path.is_file() {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    let ext = ext.to_lowercase();
                    if supported_extensions.contains(&ext.as_str()) {
                        image_paths.push(path.to_path_buf());
                    }
                }
            }
        }

        // 检查至少存在一张图片
        if image_paths.is_empty() {
            return Err(candle_core::Error::msg("no image found"));
        }

        image_paths.sort();

        // 加载第一张图片获取基准尺寸
        let first_img = image::open(&image_paths[0])
            .map_err(|e| candle_core::Error::wrap(format!("load first frame fail: {e}")))?;
        let base_dimensions = first_img.dimensions();

        Ok(Self {
            image_paths,
            base_dimensions,
        })
    }

    /// 获取总帧数
    pub fn len(&self) -> usize {
        self.image_paths.len()
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.image_paths.is_empty()
    }
}

impl FrameLoader for ImageLoader {
    fn get_frame(
        &self,
        index: usize,
        target_size: (usize, usize),
        device: &Device,
    ) -> Result<Tensor> {
        // 检查索引有效性
        let path = self.image_paths.get(index).ok_or_else(|| {
            candle_core::Error::msg(format!("index {index} out of bound（total: {}）", self.len()))
        })?;

        // 加载图片
        let img = image::open(path)
            .map_err(|e| candle_core::Error::wrap(format!("frame load fail: {e}")))?;

        // 验证尺寸一致性
        let current_dims = img.dimensions();
        if current_dims != self.base_dimensions {
            return Err(candle_core::Error::msg(format!(
                "frame size not match first frame: current {:?} ≠ base {:?}",
                current_dims, self.base_dimensions
            )));
        }

        preprocess_image(
            &img,
            (target_size.0 as u32, target_size.1 as u32),
            &IMAGE_MEAN,
            &IMAGE_STD,
            device,
        )
    }

    fn frame_size(&self) -> (usize, usize) {
        let width = self.base_dimensions.0 as usize;
        let height = self.base_dimensions.1 as usize;
        (width, height)
    }

    fn total_frames(&self) -> usize {
        self.len()
    }
}

pub struct ImaeTensorLoader {
    images: Tensor,
    base_dimensions: (u32, u32),
}

impl ImaeTensorLoader {
    pub fn new<P: AsRef<Path>>(safetensors_path: P, device: &Device) -> Result<Self> {
        let (images, height, width) = load_video_frames(safetensors_path, device)?;

        Ok(Self {
            images,
            base_dimensions: (height as u32, width as u32),
        })
    }
}

impl FrameLoader for ImaeTensorLoader {
    fn get_frame(
        &self,
        frame_idx: usize,
        _target_size: (usize, usize),
        device: &Device,
    ) -> Result<Tensor> {
        self.images.get(frame_idx)?.to_device(device)?.unsqueeze(0)
    }

    fn frame_size(&self) -> (usize, usize) {
        let width = self.base_dimensions.0 as usize;
        let height = self.base_dimensions.1 as usize;
        (width, height)
    }

    fn total_frames(&self) -> usize {
        self.images.dim(0).unwrap()
    }
}

pub fn load_video_frames<P: AsRef<std::path::Path>>(
    video_path: P,
    device: &Device,
) -> Result<(Tensor, usize, usize)> {
    let tm = candle_core::safetensors::load(video_path, device)?;

    let images = tm.get("images").unwrap().clone();
    let video_width = tm
        .get(&format!("video_width"))
        .and_then(|t| t.to_scalar::<i64>().ok())
        .unwrap() as usize;
    let video_height = tm
        .get(&format!("video_height"))
        .and_then(|t| t.to_scalar::<i64>().ok())
        .unwrap() as usize;

    Ok((images, video_height, video_width))
}
