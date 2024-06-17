use candle_core::{CpuStorage, CustomOp1, DType, Device, Layout, Shape, CudaStorage, Tensor};
use candle_core::cuda_backend::{WrapErr, CudaStorageSlice, CudaError};
use candle_core::Result as CandleResult;

use cudarc::driver::{LaunchAsync, LaunchConfig, CudaFunction};

pub fn bilinear_interpolate(
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
pub struct BilinearInterpolation {
    cuda_kernel: Option<CudaFunction>,
}

impl BilinearInterpolation {
    pub fn new(device: Device) -> Result<Self, String> {
        match device {
            Device::Cuda(device) => {
                let kernel_code = r#"

                extern "C" __global__ void upsample_bilinear2d(const float* input, float* output, int width, int height, int channels) {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    int out_width = 2 * width;
                    int out_height = 2 * height;
                
                    if (idx < channels * out_width * out_height) {
                        int c = idx / (out_width * out_height);
                        int out_y = (idx % (out_width * out_height)) / out_width;
                        int out_x = idx % out_width;
                        int in_x = out_x / 2;
                        int in_y = out_y / 2;
                        int in_idx = c * width * height + in_y * width + in_x;
                        int out_idx = c * out_width * out_height + out_y * out_width + out_x;
                
                        float x_diff = (out_x % 2 == 0) ? 0.0f : 0.5f;
                        float y_diff = (out_y % 2 == 0) ? 0.0f : 0.5f;
                
                        float top_left = input[in_idx];
                        float top_right = (in_x + 1 < width) ? input[in_idx + 1] : top_left;
                        float bottom_left = (in_y + 1 < height) ? input[in_idx + width] : top_left;
                        float bottom_right = (in_x + 1 < width && in_y + 1 < height) ? input[in_idx + width + 1] : bottom_left;
                
                        output[out_idx] = top_left * (1.0 - x_diff) * (1.0 - y_diff) +
                                        top_right * x_diff * (1.0 - y_diff) +
                                        bottom_left * (1.0 - x_diff) * y_diff +
                                        bottom_right * x_diff * y_diff;
                    }
                }

                "#;

                let ptx = cudarc::nvrtc::compile_ptx(kernel_code).unwrap();
                device.load_ptx(ptx, "upsample_bilinear2d", &["upsample_bilinear2d"]).unwrap();
                let kernel = device.get_func("upsample_bilinear2d", "upsample_bilinear2d").unwrap();

                Ok(Self {
                    cuda_kernel: Some(kernel),
                })
            }
            Device::Cpu => Ok(Self {
                cuda_kernel: None,
            }),
            _ => Err("Unsupported Device".to_string()),
        }
    }
}

impl CustomOp1 for BilinearInterpolation {
    fn name(&self) -> &'static str {
        "2x_bilinear_interpolation"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> CandleResult<(CpuStorage, Shape)>{
        Err(CudaError::InternalError("no cpu").into())
    }

    fn cuda_fwd(&self, storage: &CudaStorage, l: &Layout) -> CandleResult<(CudaStorage, Shape)> {
        let s = &storage.slice;
        let shape = l.shape().dims4().unwrap(); // [b, c, w, h]
        let size = shape.1 * shape.2*2 * shape.3*2;
        
        match s {
            CudaStorageSlice::F32(s) => {
                let kernel = self.cuda_kernel.clone();
                match kernel {
                    Some(kernel) => {
                        let cfg = LaunchConfig::for_num_elems(size as u32);
                        let out = unsafe { storage.device.alloc::<f32>(size) }.w().unwrap();

                        let dims = l.shape().dims();
                        let ds = if dims.len() == 4 {
                            [dims, l.stride()].concat()
                        } else {
                            panic!("Not an image"); // TODO replace with error, move to original shape calcs at start
                        };

                        let _ds = storage.device.htod_copy(ds).w()?;
                        let inp = s.slice(l.start_offset()..);

                        let params = (&inp, &out, shape.2 as i32, shape.3 as i32, shape.1 as i32);
                        
                        unsafe { kernel.launch(cfg, params) }.w().unwrap();
                        
                        let out = CudaStorageSlice::F32(out);
                        let out = CudaStorage { slice: out, device: storage.device.clone() };

                        Ok((out, Shape::from_dims(&[shape.0, shape.1, shape.2*2, shape.3*2])))
                    }
                    None => Err(CudaError::InternalError("Kernel").into()),
                }
            }
            _ => Err(CudaError::UnexpectedDType {
                msg: "where conditions should be f32",
                expected: DType::F32,
                got: DType::F32,
            })
            .w()?,
        }
    }
}