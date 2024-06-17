use candle_core::shape::ShapeWithOneHole;

use candle_core::{Device, Result, Tensor};
use candle_nn::{Conv2d, conv2d, Conv2dConfig, VarBuilder, Module, layer_norm, linear, LayerNorm, Linear};

use std::fmt;

use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;

use crate::models::lrelu::LRelu;
use crate::models::bilinear::BilinearInterpolation;

#[derive(Clone)]
pub struct NextConfig {
    pub num_feat: usize,
    pub num_grow_ch: usize,
    pub num_in_ch: usize,
    pub num_out_ch: usize,
    pub num_block: usize,
}

#[derive(Clone)]
struct LayerNormCL {
    norm: LayerNorm
}

impl LayerNormCL {
    pub fn load(vb: &VarBuilder, prefix: &str, config: &NextConfig) -> LayerNormCL {
        let norm = layer_norm(config.num_feat, 1e-6, vb.pp(prefix)).unwrap();
        return Self {
            norm
        }
    }

    pub fn forward(&self, x:&Tensor) -> Result<Tensor> {
        return x.apply(&self.norm);
    }
}

#[derive(Clone)]
struct GRN {
    gamma: Tensor,
    beta: Tensor,
    spatial_dim: [usize; 2],
    channel_dim: usize
}

impl GRN {
    pub fn load(vb: &VarBuilder, prefix: &str, num_dim: usize, channels_last: bool) -> GRN {

        let (shape, spatial_dim, channel_dim) = if channels_last {
            ((1, 1, 1, ()).into_shape(num_dim).unwrap(), [1, 2], 3)
        } else {
            ((1, (), 1, 1).into_shape(num_dim).unwrap(), [2, 3], 1)
        };

        let gamma = vb.get((1, 1, 1, num_dim), &format!("{}.gamma", prefix)).unwrap().reshape(&shape).unwrap();
        let beta = vb.get((1, 1, 1, num_dim), &format!("{}.beta", prefix)).unwrap().reshape(&shape).unwrap();
        return Self {
            gamma,
            beta,
            spatial_dim,
            channel_dim
        }
    }

    pub fn forward(&self, xs:&Tensor) -> Result<Tensor> {
        let residual = xs;
        let gx = xs
            .sqr()?
            .sum_keepdim(self.spatial_dim)?
            .mean_keepdim(self.spatial_dim)?
            .sqrt()?;

        let gxmean = gx.mean_keepdim(self.channel_dim)?;
        let nx = gx.broadcast_div(&(gxmean + 1e-6)?)?;
        let xs = xs
            .broadcast_mul(&nx)?
            .broadcast_mul(&self.gamma)?
            .broadcast_add(&self.beta)?;

        xs + residual
    }
}



#[derive(Clone)]
struct Block {
    dwconv: Conv2d,
    norm: LayerNormCL,
    pwconv1: Linear,
    grn: GRN,
    pwconv2: Linear,
}

impl Block {
    pub fn load(vb: &VarBuilder, prefix: &str, config: &NextConfig) -> Block {
        let conv2d_cfg = Conv2dConfig {
            groups: config.num_feat,
            padding: 3,
            ..Default::default()
        };
    
        let dwconv = conv2d(config.num_feat, config.num_feat, 7, conv2d_cfg, vb.pp(format!("{}.dwconv", prefix))).unwrap();
        let norm = LayerNormCL::load(vb, &format!("{}.norm", prefix), config);
        let pwconv1 = linear(config.num_feat, 4 * config.num_feat, vb.pp(format!("{}.pwconv1", prefix))).unwrap();
        // Normally the GELU act would be here
        let grn = GRN::load(vb, &format!("{}.grn", prefix), config.num_feat * 4, true);
        let pwconv2 = linear(4 * config.num_feat, config.num_feat, vb.pp(format!("{}.pwconv2", prefix))).unwrap();
        
        Self {
            dwconv,
            norm,
            pwconv1,
            grn,
            pwconv2
        }
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {

        // Performance: Looks like the "channel last" trick is slower here?

        let mut xs = self.dwconv.forward(&xs).unwrap();
        xs = xs.permute((0, 2, 3, 1)).unwrap();
        xs = self.norm.forward(&xs).unwrap();
        xs = self.pwconv1.forward(&xs).unwrap();
        xs = xs.gelu_erf().unwrap();
        xs = self.grn.forward(&xs).unwrap();
        xs = self.pwconv2.forward(&xs).unwrap();
        xs = xs.permute((0, 3, 1, 2)).unwrap();
        return Ok(xs)
    }
}

#[test]
fn test_block() {
    let device = Device::new_cuda(0).unwrap();

    let var_map = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &device);

    let config = NextConfig {
        num_feat: 64,
        num_grow_ch: 32,
        num_in_ch: 3,
        num_out_ch: 3,
        num_block: 6
    };

    let block = Block::load(&vb, "block", &config);

    // Warmup
    for _i in 0..10 {
        let x0 = Tensor::zeros((1, 64, 64, 64), candle_core::DType::F32, &device).unwrap();
        let _ = block.forward(&x0);
    }

    let x0 = Tensor::zeros((1, 64, 64, 64), candle_core::DType::F32, &device).unwrap();
    let start = std::time::Instant::now();
    let _ = block.forward(&x0);

    println!("Block {:?}", start.elapsed());
}


#[derive(Clone)]
struct ResidualDenseBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv4: Conv2d,
    conv5: Conv2d,
    lrelu: LRelu
}

impl ResidualDenseBlock {
    pub fn load(vb: &VarBuilder, prefix: &str, config: &NextConfig, device:&Device) -> ResidualDenseBlock {
        let c = Conv2dConfig {stride: 1, padding: 1, dilation: 1, groups: 1};
        let conv1 = conv2d(config.num_feat + 0 * config.num_grow_ch, config.num_grow_ch, 3, c, vb.pp(&format!("{}.conv1", prefix))).unwrap();
        let conv2 = conv2d(config.num_feat + 1 * config.num_grow_ch, config.num_grow_ch, 3, c, vb.pp(&format!("{}.conv2", prefix))).unwrap();
        let conv3 = conv2d(config.num_feat + 2 * config.num_grow_ch, config.num_grow_ch, 3, c, vb.pp(&format!("{}.conv3", prefix))).unwrap();
        let conv4 = conv2d(config.num_feat + 3 * config.num_grow_ch, config.num_grow_ch, 3, c, vb.pp(&format!("{}.conv4", prefix))).unwrap();
        let conv5 = conv2d(config.num_feat + 4 * config.num_grow_ch, config.num_feat, 3, c, vb.pp(&format!("{}.conv5", prefix))).unwrap();
        let lrelu = LRelu::new(0.2, device.clone()).unwrap();
        Self { conv1, conv2, conv3, conv4, conv5, lrelu}
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
    config: NextConfig,
    block: Block,
    l1: Linear,
    rdb1: ResidualDenseBlock,
    rdb2: ResidualDenseBlock,
    rdb3: ResidualDenseBlock,
}

impl RRDB {
    pub fn load(vb: &VarBuilder, prefix: &str, config: &NextConfig, device:&Device) -> RRDB {

        let block = Block::load(vb, &format!("{}.block", prefix), config);
        let l1 = linear(config.num_feat, config.num_feat, vb.pp(format!("{}.l1", prefix))).unwrap();
        let rdb1 = ResidualDenseBlock::load(vb, &format!("{}.rdb1", prefix), config, device);
        let rdb2 = ResidualDenseBlock::load(vb, &format!("{}.rdb2", prefix), config, device);
        let rdb3 = ResidualDenseBlock::load(vb, &format!("{}.rdb3", prefix), config, device);
        let config = config.clone();
        Self { config, block, l1, rdb1, rdb2, rdb3 }
    }

    pub fn forward(&self, x: &Tensor, c:&Tensor, scale:&Tensor) -> Result<Tensor> {
        
        let condition = self.l1.forward(&c).unwrap();
        let (batch, _, _, _) = x.dims4().unwrap();

        // TODO - Incorrect batch calculations, need to repeat (batch=1 should work fine)
        let condition = condition.reshape((batch, self.config.num_feat, 1, 1)).unwrap();

        let x_mixed = x.broadcast_add(&condition).unwrap();
        let features = self.block.forward(&x_mixed).unwrap();
        let features = features.broadcast_mul(&scale).unwrap();
        
        let out = self.rdb1.forward(&(x + &features)?)?;
        let out = self.rdb2.forward(&(out + &features)?)?;
        let out = self.rdb3.forward(&(out + &features)?)?;
        out * 0.2 + x
    }
}

#[derive(Deserialize, Clone)]
struct CondData {
    average: Vec<f32>,
    detail: Vec<f32>,
    spikes: Vec<f32>,
    stars: Vec<f32>,
}

#[derive(Clone)]
struct TensorCondData {
    average: Tensor,
    detail: Tensor,
    spikes: Tensor,
    stars: Tensor,
}


#[derive(Clone)]
pub struct Next {
    config: NextConfig,
    l1: Linear,
    conv_first: Conv2d,
    body: Vec<RRDB>,
    conv_body: Conv2d,
    conv_up1: Conv2d,
    conv_up2: Conv2d,
    conv_hr: Conv2d,
    conv_last: Conv2d,
    cond_data: TensorCondData,
    lrelu: LRelu,
    up: BilinearInterpolation,
    pub device: Device,
}

impl fmt::Debug for Next {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Next").field("Device", &self.device).finish()
    }
}

impl Next {
    pub fn load(vb: &VarBuilder, config: &NextConfig) -> Result<Next> {
        let c = Conv2dConfig {stride: 1, padding: 1, dilation: 1, groups: 1};
        let device = vb.pp("conv_first").device().clone();

        let l1 = linear(512, config.num_feat, vb.pp("l1")).unwrap();
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
        let up = BilinearInterpolation::new(device.clone()).unwrap();

        let cond_data = Next::load_json();
        let cond_data = TensorCondData{
            average: Tensor::from_vec(cond_data.average.clone(), (1, 512), &device).unwrap(),
            detail: Tensor::from_vec(cond_data.detail.clone(), (1, 512), &device).unwrap(),
            spikes: Tensor::from_vec(cond_data.spikes.clone(), (1, 512), &device).unwrap(),
            stars: Tensor::from_vec(cond_data.stars.clone(), (1, 512), &device).unwrap(),
        };

        let config = config.clone();

        Ok(Self {config, l1, conv_first, body, conv_body, conv_up1, conv_up2, conv_hr, conv_last, cond_data, lrelu, up, device})
    }

    fn load_json() -> CondData {
        let file = File::open("conds.json").unwrap();
        let reader = BufReader::new(file);
        let data: CondData = serde_json::from_reader(reader).unwrap();
        return data
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let cond = self.cond_data.average.clone();
        let cond = self.l1.forward(&cond).unwrap().reshape((1, 64, 1, 1)).unwrap().apply_op1_no_bwd(&self.lrelu)?.reshape((1, 64)).unwrap();
        
        let scale = Tensor::from_vec(vec![(0.0) as f32], (1, 1, 1, 1), x.device()).unwrap().repeat((1, self.config.num_feat, 1, 1)).unwrap();
        let mut feat = self.conv_first.forward(&x)?;
        for block in self.body.iter() {
            feat = block.forward(&feat, &cond, &scale).unwrap();
        }
        feat = (&feat + self.conv_body.forward(&feat)?).unwrap();

        feat = feat.apply_op1_no_bwd(&self.up)?;
        feat = self.conv_up1.forward(&feat).unwrap().apply_op1_no_bwd(&self.lrelu)?;

        feat = feat.apply_op1_no_bwd(&self.up)?;
        feat = self.conv_up2.forward(&feat).unwrap().apply_op1_no_bwd(&self.lrelu)?;
        self.conv_last.forward(&self.conv_hr.forward(&feat)?.apply_op1_no_bwd(&self.lrelu)?)
    }

    
}

