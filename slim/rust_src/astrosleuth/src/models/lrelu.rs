use candle_core::{cpu_backend, CpuStorage, CudaStorage, CustomOp1, DType, Device, Error, Layout, Shape};

use candle_core::cuda_backend::{WrapErr, CudaStorageSlice, CudaError};
use candle_core::backend::BackendStorage;

use candle_core::Result as CandleResult;

use cudarc::driver::{CudaSlice, LaunchAsync, LaunchConfig, CudaFunction};
use num_traits;

fn fwd<T: num_traits::Float>(v: T, alpha: f64) -> T {
    if v.is_sign_positive() {
        v
    } else {
        let alpha = T::from(alpha).unwrap_or(T::nan());
        v * alpha
    }
}

#[derive(Clone)]
pub struct LRelu {
    alpha: f64,
    cuda_kernel: Option<CudaFunction>,
}

impl LRelu {
    pub fn new(alpha: f64, device: Device) -> Result<Self, String> {
        match device {
            Device::Cuda(device) => {
                let kernel_code = r#"
                    extern "C" __global__ void lrelu_kernel(float* out, const float* in, float alpha, int size) {
                        int idx = blockIdx.x * blockDim.x + threadIdx.x;
                        if (idx < size) {
                            float v = in[idx];
                            out[idx] = (v > 0.0) ? v : alpha * v;
                        }
                    }
                "#;

                let ptx = cudarc::nvrtc::compile_ptx(kernel_code).unwrap();
                device.load_ptx(ptx, "lrelu", &["lrelu_kernel"]).unwrap();
                let kernel = device.get_func("lrelu", "lrelu_kernel").unwrap();

                Ok(Self {
                    alpha,
                    cuda_kernel: Some(kernel),
                })
            }
            Device::Cpu => Ok(Self {
                alpha,
                cuda_kernel: None,
            }),
            _ => Err("Unsupported Device".to_string()),
        }
    }
}

impl CustomOp1 for LRelu {
    fn name(&self) -> &'static str {
        "lrelu"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> CandleResult<(CpuStorage, Shape)> {
        let storage = candle_core::map_dtype!(
            "lrelu",
            s,
            |s| cpu_backend::unary_map(s, l, |v| fwd(v, self.alpha)),
            (BF16, F16, F32, F64)
        );
        Ok((storage, l.shape().clone()))
    }

    fn cuda_fwd(&self, storage: &CudaStorage, l: &Layout) -> CandleResult<(CudaStorage, Shape)> {
        let alpha = self.alpha as f32;
        let s = &storage.slice;
        let shape = l.shape().dims4().unwrap(); // [b, c, w, h]
        let size = shape.0 * shape.1 * shape.2 * shape.3;
        
        match s {
            CudaStorageSlice::F32(s) => {
                let kernel = self.cuda_kernel.clone();
                match kernel {
                    Some(kernel) => {
                        let cfg = LaunchConfig::for_num_elems(size as u32);
                        let out: CudaSlice<f32> = unsafe { storage.device.alloc::<f32>(size) }.w().unwrap();
                        
                        let inp = s.slice(l.start_offset()..);
                        let params = (&out, &inp, alpha as f32, size as i32);

                        unsafe { kernel.launch(cfg, params) }.w().unwrap();
                        
                        let out = CudaStorageSlice::F32(out);
                        let out = CudaStorage { slice: out, device: storage.device.clone() };

                        Ok((out, l.shape().clone()))
                    }
                    None => Err(CudaError::InternalError("Kernel").into()),
                }
            }
            _ => Err(CudaError::UnexpectedDType {
                msg: "where conditions should be f32",
                expected: DType::F32,
                got: DType::F32, // TODO, find input dtype?
            })
            .w()?,
        }
    }

}