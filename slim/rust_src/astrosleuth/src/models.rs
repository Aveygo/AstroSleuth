
use candle_core::{Device, Tensor, Error, DType};
use candle_nn::VarBuilder;
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::Result as CandleResult;

use std::io::{Write,stdout};

use image::{DynamicImage, GenericImageView, Rgba, GenericImage};
use image;
use core::str::FromStr;
use std::time::Instant;

pub mod realesr;
pub mod srvggish;
pub mod next;
pub mod lrelu;
pub mod bilinear;



#[derive(Debug, Clone)]
pub enum Model {
    RealESRGAN(realesr::RealESRGAN),
    SRVGGISH(srvggish::SRVGGISH),
    Next(next::Next)
}

pub fn device(cpu: bool) -> CandleResult<Device> {
    if !cpu {
        if cuda_is_available() {
            println!("Using the GPU");
            return Ok(Device::new_cuda(0)?)
        } else if metal_is_available() {
            println!("Using Apple Metal Acceleration");
            return Ok(Device::new_metal(0)?)
        }
    } 
    println!("Running on CPU (No Acceleration)");
    Ok(Device::Cpu) // Fallback to cpu
}

impl FromStr for Model {

    type Err = String;

    fn from_str(input: &str) -> Result<Model, String> {
        match input.to_lowercase().as_str() {
            "realesrgan" => {
                let config = realesr::RealESRGANConfig {
                    num_feat: 64,
                    num_grow_ch: 32,
                    num_in_ch: 3,
                    num_out_ch: 3,
                    num_block: 6,
                };

                let vb = unsafe { 
                    VarBuilder::from_mmaped_safetensors(&["models/astrosleuthv1.safetensors"], DType::F32, &device(false).unwrap())
                        .map_err(|e| e.to_string()).unwrap() 
                };

                let model = realesr::RealESRGAN::load(&vb, &config).map_err(|e| e.to_string()).unwrap();
                Ok(Model::RealESRGAN(model))
            },
            "astrosleuthfast" => {
                let config = srvggish::SRVGGISHConfig {
                    num_feat: 64,
                    num_conv: 32,
                    num_in_ch: 3,
                    num_out_ch: 3,
                };

                let vb = unsafe { 
                    VarBuilder::from_mmaped_safetensors(&["models/astrosleuthfast.safetensors"], DType::F32, &device(false).unwrap())
                        .map_err(|e| e.to_string()).unwrap() 
                };

                let model = srvggish::SRVGGISH::load(&vb, &config).map_err(|e| e.to_string()).unwrap();
                Ok(Model::SRVGGISH(model))
            },
            "next" => {
                let config = next::NextConfig {
                    num_feat: 64,
                    num_grow_ch: 32,
                    num_in_ch: 3,
                    num_out_ch: 3,
                    num_block: 6,
                };

                let vb = unsafe { 
                    VarBuilder::from_mmaped_safetensors(&["models/next.safetensors"], DType::F32, &device(false).unwrap())
                        .map_err(|e| e.to_string()).unwrap() 
                };

                let model = next::Next::load(&vb, &config).map_err(|e| e.to_string()).unwrap();
                Ok(Model::Next(model))
            },


            _ => Err(format!("Invalid model: {}", input)),
        }
    }
}

pub struct Worker<'a> {
    tile_size: u32,
    tile_pad: u32,
    scale: u32,
    model: &'a Model,
}
struct WorkerResultTile {
    data: DynamicImage,
    x: u32,
    y: u32
}

impl<'a> Worker<'a> {
    pub fn new(tile_size:u32, tile_pad:u32, scale:u32, model: &'a Model) -> Worker {
        Self {
            tile_size: tile_size,
            tile_pad: tile_pad,
            scale: scale,
            model: model
        }
    }

    pub fn add_padding(&self,image: &DynamicImage, padding: u32) -> DynamicImage {
        let (width, height) = image.dimensions();
        let new_width = width + 2 * padding;
        let new_height = height + 2 * padding;
    
        // Create a new image with the desired size and a black background (or any color you prefer)
        let mut padded_image = DynamicImage::new_rgb8(new_width, new_height);
        for x in 0..new_width {
            for y in 0..new_height {
                padded_image.put_pixel(x, y, Rgba([0, 0, 0, 0]));
            }
        }
    
        // Copy the original image into the center of the new image
        padded_image.copy_from(image, padding, padding).unwrap();
    
        padded_image
    }

    pub fn upscale<P: AsRef<std::path::Path>>(&self, src: &P, dst: &P) {
        let now = Instant::now();

        let mut image = image::io::Reader::open(src).unwrap().decode().unwrap();
        let (w, h) = image.dimensions();
        
        // Ensure that the image can fit each tile
        let initial_w = w + self.tile_size - 1 - (w + self.tile_size - 1) % self.tile_size;
        let initial_h = h + self.tile_size - 1 - (h + self.tile_size - 1) % self.tile_size;
        image = image.resize_exact(initial_w, initial_h, image::imageops::FilterType::Lanczos3);

        let slippage = 1. - (initial_w*initial_h) as f32 / (w*h) as f32;
        println!("Started [{:?}x{:?}] -> [{:?}x{:?}] ({:.2}% match)", w, h, w*4, h*4, slippage);

        // Add padding to ensure square tiles
        let image = self.add_padding(&image, self.tile_pad);

        let mut result = self.process_tiles(&image);

        // remove padding
        let mut result = result.crop(
            self.tile_pad * 4, 
            self.tile_pad * 4, 
            initial_w * 4, 
            initial_h * 4
        );


        result = result.resize_exact(w*4, h*4, image::imageops::FilterType::Lanczos3);
        result.save(dst).unwrap();
        println!("Job took {:?} seconds", now.elapsed().as_secs());
    }

    pub fn inference(&self, img: DynamicImage) -> DynamicImage {

        let (w, h) = img.dimensions();
        // println!("{:?}, {:?}", w, h);

        let img = img.to_rgb8();
        let x = img.into_raw();
        let x = Tensor::from_vec(x, (h as usize, w as usize, 3 as usize), &Device::Cpu).unwrap().permute((2, 0, 1)).unwrap();
        let x = (x.to_dtype(DType::F32).unwrap() / 255.).unwrap().unsqueeze(0).unwrap();

        let mut y;
        match &self.model {
            Model::RealESRGAN(model) => {
                y = model.forward(&x.to_device(&model.device).unwrap()).unwrap();
            },
            Model::SRVGGISH(model) => {
                y = model.forward(&x.to_device(&model.device).unwrap()).unwrap();
            },
            Model::Next(model) => {
                y = model.forward(&x.to_device(&model.device).unwrap()).unwrap();
            }
        }
        
        y = y.squeeze(0).unwrap().clamp(0.0, 1.0).unwrap();
        y = (y * 255.).unwrap().to_dtype(DType::U8).unwrap().clamp(0., 255.).unwrap();
        let (_, height, width) = y.dims3().unwrap();
        let img = y.permute((1, 2, 0)).unwrap().flatten_all().unwrap();
        let pixels = img.to_vec1::<u8>().unwrap();
        let image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            match image::ImageBuffer::from_raw(width as u32, height as u32, pixels) {
                Some(image) => image,
                None => panic!("error converting image!"),
            };

        return image::DynamicImage::ImageRgb8(image);
    }


    pub fn process_tiles(&self, image: &DynamicImage) -> DynamicImage {
        let (w, h) = image.dimensions();
        let mut result = DynamicImage::new_rgb8(w*4, h*4);
        let num_tiles = self.num_tiles(image);
        println!("Total tiles: {:?}", num_tiles);

        for tile_id in 0..num_tiles {
            self.paste_tile(&mut result, self.get_tile_data(image, tile_id));
            print!("{:.2}%\r", (tile_id as f32 / num_tiles as f32) * 100.);
            let _ = stdout().flush();
        }
        print!("100.00%\n");
        let _ = stdout().flush();

        return result;
    }

    pub fn _load_image<P: AsRef<std::path::Path>>(p: P) -> CandleResult<Tensor> {
        let img = image::io::Reader::open(p)?
            .decode()
            .map_err(Error::wrap)?;
        let img = img.to_rgb8();
        let data = img.into_raw();
        let data = Tensor::from_vec(data, (64, 64, 3), &Device::Cpu)?.permute((2, 0, 1))?;
        return data.to_dtype(DType::F32)? / 255.
    }
    
    pub fn _save_image<P: AsRef<std::path::Path>>(img: &Tensor, p: P) -> CandleResult<()> {
        let p = p.as_ref();
        let (channel, height, width) = img.dims3()?;
        if channel != 3 {
            panic!("save_image expects an input of shape (3, height, width)")
        }
        let img = img.permute((1, 2, 0))?.flatten_all()?;
        let pixels = img.to_vec1::<u8>()?;
        let image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            match image::ImageBuffer::from_raw(width as u32, height as u32, pixels) {
                Some(image) => image,
                None => panic!("error saving image {p:?}"),
            };
        image.save(p).map_err(Error::wrap)?;
        Ok(())
    }

    fn get_tile_data(
        &self,
        image: &DynamicImage,
        tile_id: u32,
    ) -> WorkerResultTile {
        let (width, height) = image.dimensions();

        let tiles_x = width / self.tile_size;
        let tiles_y = height / self.tile_size;
    
        if tile_id >= tiles_y * tiles_x {
            panic!("Tile ID exceeds number of tiles");
        }
    
        let x = tile_id % tiles_y;
        let y = tile_id / tiles_y;

        let input_start_x = y * self.tile_size + self.tile_pad;
        let input_start_y = x * self.tile_size + self.tile_pad;
    
        let input_end_x = (input_start_x + self.tile_size).min(width);
        let input_end_y = (input_start_y + self.tile_size).min(height);
    
        let input_start_x_pad = input_start_x.saturating_sub(self.tile_pad);
        let input_end_x_pad = (input_end_x + self.tile_pad).min(width);
        let input_start_y_pad = input_start_y.saturating_sub(self.tile_pad);
        let input_end_y_pad = (input_end_y + self.tile_pad).min(height);
    
        let input_tile = image.clone().crop(
            input_start_x_pad, 
            input_start_y_pad, 
            input_end_x_pad - input_start_x_pad, 
            input_end_y_pad - input_start_y_pad
        );
    
        let mut output_tile = self.inference(input_tile);
    
        let output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale;
        let output_end_x_tile = output_start_x_tile + (input_end_x - input_start_x) * self.scale;
        let output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale;
        let output_end_y_tile = output_start_y_tile + (input_end_y - input_start_y) * self.scale;

        let result = output_tile.crop(
            output_start_x_tile, 
            output_start_y_tile, 
            output_end_x_tile - output_start_x_tile, 
            output_end_y_tile - output_start_y_tile
        );

        return WorkerResultTile {
            data: result,
            x: input_start_x,
            y: input_start_y
        }

    }

    fn paste_tile(&self, base_img: &mut DynamicImage, tile:WorkerResultTile) {
        let base_img = base_img.as_mut_rgb8().expect("Base image must be RGB8");
        let overlay_img = tile.data.to_rgb8();
        let (overlay_width, overlay_height) = overlay_img.dimensions();
    
        for ox in 0..overlay_width {
            for oy in 0..overlay_height {
                let bx = tile.x * 4 + ox;
                let by = tile.y * 4 + oy;
    
                if bx >= base_img.width() || by >= base_img.height() {
                    continue;
                }
    
                let pixel = overlay_img.get_pixel(ox, oy);
    
                base_img.put_pixel(bx, by, *pixel);
            }
        }
    }

    pub fn num_tiles(&self, image: &DynamicImage) -> u32 {
        let (width, height) = image.dimensions();
    
        let tiles_x = width / self.tile_size;
        let tiles_y = height / self.tile_size;
        return tiles_y * tiles_x
    }

    
    
}