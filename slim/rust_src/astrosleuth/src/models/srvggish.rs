use candle_core::{Result, Tensor, Device};
use candle_nn::{conv2d, prelu, Conv2d, Conv2dConfig, Module, PReLU, VarBuilder};
use std::fmt;

pub struct SRVGGISHConfig {
    pub num_feat: usize,
    pub num_conv: usize,
    pub num_in_ch: usize,
    pub num_out_ch: usize,
}

struct LRelu;

impl LRelu {
    fn forward(input: &Tensor) -> Tensor {
        let original_shape = input.shape();
        let flat = input.flatten_all().unwrap().unsqueeze(0).unwrap();
        let zero = flat.zeros_like().unwrap();
        let joined = Tensor::cat(&[zero, flat], 0).unwrap();
        let added = (joined.min(0).unwrap() * 0.2 + joined.max(0).unwrap()).unwrap();
        added.reshape(original_shape).unwrap()
    }
}
fn bilinear_interpolate(
    input_tensor: &Tensor,
    new_width: usize,
    new_height: usize
) -> Tensor {

    let (batch, channel, width, height) = input_tensor.shape().dims4().unwrap();
    let input: Vec<f32> = input_tensor.flatten_all().unwrap().to_vec1().unwrap();

    let mut output = vec![0.0; batch * channel * new_width * new_height];

    // Handle cases where new_width or new_height is 1 to avoid division by zero
    let scale_x = if new_width > 1 {
        (width - 1) as f32 / (new_width - 1) as f32
    } else {
        0.0
    };

    let scale_y = if new_height > 1 {
        (height - 1) as f32 / (new_height - 1) as f32
    } else {
        0.0
    };

    for b in 0..batch {
        for c in 0..channel {
            for new_x in 0..new_width {
                for new_y in 0..new_height {
                    let x = new_x as f32 * scale_x;
                    let y = new_y as f32 * scale_y;

                    let x0 = x.floor() as usize;
                    let x1 = (x0 + 1).min(width - 1);
                    let y0 = y.floor() as usize;
                    let y1 = (y0 + 1).min(height - 1);

                    let p00 = input[(b * channel * width * height) + (c * width * height) + (y0 * width) + x0];
                    let p01 = input[(b * channel * width * height) + (c * width * height) + (y1 * width) + x0];
                    let p10 = input[(b * channel * width * height) + (c * width * height) + (y0 * width) + x1];
                    let p11 = input[(b * channel * width * height) + (c * width * height) + (y1 * width) + x1];

                    let dx = x - x0 as f32;
                    let dy = y - y0 as f32;

                    let interpolated = p00 * (1.0 - dx) * (1.0 - dy)
                                    + p10 * dx * (1.0 - dy)
                                    + p01 * (1.0 - dx) * dy
                                    + p11 * dx * dy;

                    output[(b * channel * new_width * new_height) + (c * new_width * new_height) + (new_y * new_width) + new_x] = interpolated;
                }
            }
        }
    }

    let tensor_output = Tensor::from_vec(output, (batch, channel, new_width, new_height), input_tensor.device()).unwrap();

    tensor_output
}

#[derive(Clone)]
pub struct SRVGGISH {
    body: Vec<BodyItem>,
    conv_up1: Conv2d,
    conv_up2: Conv2d,
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
        Ok(Self {body, conv_up1, conv_up2, device})
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {

        let skip = x.clone();

        let mut x = self.body.iter()
            .fold(x.clone(), |acc, block| block.forward(&acc).unwrap());
        
        let new_width = x.shape().dims().get(2).unwrap() * 2;
        let new_height = x.shape().dims().get(3).unwrap() * 2;
        x = bilinear_interpolate(&x, new_width, new_height);
        x = LRelu::forward(&self.conv_up1.forward(&x).unwrap());

        let new_width = x.shape().dims().get(2).unwrap() * 2;
        let new_height = x.shape().dims().get(3).unwrap() * 2;
        x = bilinear_interpolate(&x, new_width, new_height);
        x = self.conv_up2.forward(&x).unwrap();

        return x + skip.upsample_nearest2d(
            *skip.shape().dims().get(2).unwrap() * 4,
            *skip.shape().dims().get(3).unwrap() * 4,
        )?
        
    }

    
}

