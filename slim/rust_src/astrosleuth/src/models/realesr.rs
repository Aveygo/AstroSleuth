use candle_core::{Result, Tensor, Device};
use candle_nn::{Conv2d, conv2d, Conv2dConfig, VarBuilder, Module};
use serde::Deserialize;
use std::fmt;

use crate::models::lrelu::LRelu;

#[derive(Debug, Clone, Deserialize)]
pub struct RealESRGANConfig {
    pub num_feat: usize,
    pub num_grow_ch: usize,
    pub num_in_ch: usize,
    pub num_out_ch: usize,
    pub num_block: usize,
}

#[derive(Clone)]
struct ResidualDenseBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv4: Conv2d,
    conv5: Conv2d,
    lrelu: LRelu,
}

impl ResidualDenseBlock {
    pub fn load(vb: &VarBuilder, prefix: &str, config: &RealESRGANConfig, device:&Device) -> ResidualDenseBlock {
        let c = Conv2dConfig {stride: 1, padding: 1, dilation: 1, groups: 1};
        let conv1 = conv2d(config.num_feat + 0 * config.num_grow_ch, config.num_grow_ch, 3, c, vb.pp(&format!("{}.conv1", prefix))).unwrap();
        let conv2 = conv2d(config.num_feat + 1 * config.num_grow_ch, config.num_grow_ch, 3, c, vb.pp(&format!("{}.conv2", prefix))).unwrap();
        let conv3 = conv2d(config.num_feat + 2 * config.num_grow_ch, config.num_grow_ch, 3, c, vb.pp(&format!("{}.conv3", prefix))).unwrap();
        let conv4 = conv2d(config.num_feat + 3 * config.num_grow_ch, config.num_grow_ch, 3, c, vb.pp(&format!("{}.conv4", prefix))).unwrap();
        let conv5 = conv2d(config.num_feat + 4 * config.num_grow_ch, config.num_feat, 3, c, vb.pp(&format!("{}.conv5", prefix))).unwrap();
        let lrelu = LRelu::new(0.2, device.clone()).unwrap();
        Self { conv1, conv2, conv3, conv4, conv5, lrelu }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x1 = &self.conv1.forward(&x)?.apply_op1_no_bwd(&self.lrelu)?;
        let x2 = &self.conv2.forward(&Tensor::cat(&[x, &x1], 1).unwrap())?.apply_op1_no_bwd(&self.lrelu)?;
        let x3 = &self.conv3.forward(&Tensor::cat(&[x, &x1, &x2], 1).unwrap())?.apply_op1_no_bwd(&self.lrelu)?;
        let x4 = &self.conv4.forward(&Tensor::cat(&[x, &x1, &x2, &x3], 1).unwrap())?.apply_op1_no_bwd(&self.lrelu)?;
        let x5 = self.conv5.forward(&Tensor::cat(&[x, &x1, &x2, &x3, &x4], 1).unwrap())?;
        x5 * 0.2 + x
    }
}

#[derive(Clone)]
struct RRDB {
    rdb1: ResidualDenseBlock,
    rdb2: ResidualDenseBlock,
    rdb3: ResidualDenseBlock,
}

impl RRDB {
    pub fn load(vb: &VarBuilder, prefix: &str, config: &RealESRGANConfig, device:&Device) -> RRDB {
        let rdb1 = ResidualDenseBlock::load(vb, &format!("{}.rdb1", prefix), config, &device);
        let rdb2 = ResidualDenseBlock::load(vb, &format!("{}.rdb2", prefix), config, &device);
        let rdb3 = ResidualDenseBlock::load(vb, &format!("{}.rdb3", prefix), config, &device);
        Self { rdb1, rdb2, rdb3 }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.rdb1.forward(x)?;
        let out = self.rdb2.forward(&out)?;
        let out = self.rdb3.forward(&out)?;
        out * 0.2 + x
    }
}


#[derive(Clone)]
pub struct RealESRGAN {
    conv_first: Conv2d,
    body: Vec<RRDB>,
    conv_body: Conv2d,
    conv_up1: Conv2d,
    conv_up2: Conv2d,
    conv_hr: Conv2d,
    conv_last: Conv2d,
    lrelu: LRelu,
    pub device: Device,
}

impl fmt::Debug for RealESRGAN {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RealESRGAN").field("Device", &self.device).finish()
    }
}

impl RealESRGAN {
    pub fn load(vb: &VarBuilder, config: &RealESRGANConfig) -> Result<RealESRGAN> {
        let c = Conv2dConfig {stride: 1, padding: 1, dilation: 1, groups: 1};
        let device = vb.pp("conv_first").device().clone();

        let conv_first = conv2d(config.num_in_ch, config.num_feat, 3, c, vb.pp("conv_first")).unwrap();
        
        let body = (0..config.num_block)
            .map(|block_id| RRDB::load(vb, &format!("body.{}", block_id), config, &device))
            .collect();

        let conv_body = conv2d(config.num_feat, config.num_feat, 3, c, vb.pp("conv_body")).unwrap();
        let conv_up1 = conv2d(config.num_feat, config.num_feat, 3, c, vb.pp("conv_up1")).unwrap();
        let conv_up2 = conv2d(config.num_feat, config.num_feat, 3, c, vb.pp("conv_up2")).unwrap();
        let conv_hr = conv2d(config.num_feat, config.num_feat, 3, c, vb.pp("conv_hr")).unwrap();
        let conv_last = conv2d(config.num_feat, config.num_out_ch, 3, c, vb.pp("conv_last")).unwrap();

        let lrelu = LRelu::new(0.2, device.clone()).unwrap();

        Ok(Self {conv_first, body, conv_body, conv_up1, conv_up2, conv_hr, conv_last, lrelu, device})
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut feat = self.conv_first.forward(x)?;

        let body_feat = self.body.iter()
            .fold(feat.clone(), |acc, block| block.forward(&acc).unwrap());
        
        feat = (feat + self.conv_body.forward(&body_feat)?).unwrap();
        
        feat = self.conv_up1.forward(&feat.upsample_nearest2d(
            *feat.shape().dims().get(2).unwrap() * 2,
            *feat.shape().dims().get(3).unwrap() * 2,
        )?)?.apply_op1_no_bwd(&self.lrelu)?;

        feat = self.conv_up2.forward(&feat.upsample_nearest2d(
            *feat.shape().dims().get(2).unwrap() * 2,
            *feat.shape().dims().get(3).unwrap() * 2,
        )?)?.apply_op1_no_bwd(&self.lrelu)?;

        self.conv_last.forward(&self.conv_hr.forward(&feat)?.apply_op1_no_bwd(&self.lrelu)?)
    }

    
}

