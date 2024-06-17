#[cfg(test)]
use super::*;

use candle_core::{Tensor, DType, Device};
use crate::models;

extern crate criterion;
use criterion::{black_box, Criterion, BenchmarkId, criterion_group, criterion_main};

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


#[test]
fn hello_world() {
    assert_eq!(1, 1);   
}

fn bilinear_test() {
    let x = Tensor::zeros((1, 3, 512, 512), DType::F32, &Device::Cpu).unwrap();
    let y = bilinear_interpolate(&x, 1024, 1024);
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Computation Group");
    for i in 0..100 {
        group.bench_with_input(BenchmarkId::from_parameter(i), &i, |b, &i| {
            b.iter(|| bilinear_test());
        });
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);