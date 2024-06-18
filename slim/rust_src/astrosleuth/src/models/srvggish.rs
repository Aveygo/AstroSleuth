use candle_core::{Result, Tensor, Device};
use candle_nn::{conv2d, prelu, Conv2d, Conv2dConfig, Module, PReLU, VarBuilder};
use std::fmt;
use serde::Deserialize;

use crate::models::lrelu::LRelu;
use crate::models::bilinear::BilinearInterpolation;

#[derive(Debug, Clone, Deserialize)]
pub struct SRVGGISHConfig {
    pub num_feat: usize,
    pub num_conv: usize,
    pub num_in_ch: usize,
    pub num_out_ch: usize,
}


#[derive(Clone)]
pub struct SRVGGISH {
    body: Vec<BodyItem>,
    conv_up1: Conv2d,
    conv_up2: Conv2d,
    lrelu: LRelu,
    up: BilinearInterpolation,
    pub device: Device,
}

impl fmt::Debug for SRVGGISH {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SRVGGISH").field("Device", &self.device).finish()
    }
}

#[derive(Clone)]
pub enum BodyItem {
    Conv2d(Conv2d),
    PReLU(PReLU)
}

impl Module for BodyItem {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            BodyItem::Conv2d(conv2d) => conv2d.forward(input),
            BodyItem::PReLU(prelu) => prelu.forward(input),
        }
    }
}

impl SRVGGISH {
    pub fn load(vb: &VarBuilder, config: &SRVGGISHConfig) -> Result<SRVGGISH> {
        let c = Conv2dConfig {stride: 1, padding: 1, dilation: 1, groups: 1};
        let device = vb.pp("conv_up1").device().clone();

        let mut body:Vec<BodyItem> = vec![];

        body.push(BodyItem::Conv2d(conv2d(config.num_in_ch, config.num_feat, 3, c, vb.pp("body.0")).unwrap()));
        body.push(BodyItem::PReLU(prelu(config.num_feat.into(), vb.pp(format!("body.1"))).unwrap()));

        for block_id in 0..config.num_conv {
            body.push(BodyItem::Conv2d(conv2d(config.num_feat, config.num_feat, 3, c, vb.pp(format!("body.{:.}", block_id*2 + 2))).unwrap()));
            body.push(BodyItem::PReLU(prelu(config.num_feat.into(), vb.pp(format!("body.{:.}", block_id*2 + 3))).unwrap()));
        }

        let conv_up1 = conv2d(64, 32, 3, c, vb.pp("conv_up1")).unwrap();
        let conv_up2 = conv2d(32, config.num_out_ch, 3, c, vb.pp("conv_up2")).unwrap();
        
        let lrelu = LRelu::new(0.2, device.clone()).unwrap();
        let up = BilinearInterpolation::new(device.clone()).unwrap();

        Ok(Self {body, conv_up1, conv_up2, lrelu, up, device})
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {

        let skip = x.clone();

        let mut x = self.body.iter()
            .fold(x.clone(), |acc, block| block.forward(&acc).unwrap());
        
        x = self.conv_up1.forward(&x.apply_op1_no_bwd(&self.up)?).unwrap().apply_op1_no_bwd(&self.lrelu)?;
        x = self.conv_up2.forward(&x.apply_op1_no_bwd(&self.up)?).unwrap();

        return x + skip.upsample_nearest2d(
            *skip.shape().dims().get(2).unwrap() * 4,
            *skip.shape().dims().get(3).unwrap() * 4,
        )?
        
    }

    
}

