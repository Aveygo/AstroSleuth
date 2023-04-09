import gradio as gr
import math, torch, time, tifffile, cv2, numpy as np, os
from PIL import Image
from torch.nn import functional as F
from torch import nn as nn
from network import Network

class AstroSleuth():
    def __init__(self, scale, model_path, blocks=6, tile=128, tile_pad=10, pre_pad=10, device=False):
        
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.device = device
    
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
        model = Network(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=blocks, num_grow_ch=32, scale=scale)

        if model_path == "LATEST":
            state_dict = torch.hub.load_state_dict_from_url("https://huggingface.co/spaces/CodingCoda/AstroSleuth/resolve/main/model.pth", map_location=torch.device(self.device))

        elif not os.path.exists(model_path):
            raise Exception("Model not found!")

        else:
            state_dict = torch.load(model_path, map_location=torch.device(self.device))

        model.load_state_dict(state_dict, strict=True)

        if self.device == "cuda":
            model = model.half()

        model.eval()
        self.model = model.to(self.device)    

    def tile_generator(self, data):
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

        data = torch.from_numpy(data).to(self.device).transpose(2,0).unsqueeze(0) / 255
        if self.device == "cuda":
            data = data.half()
        data = data.clip(0, 1)

        batch, channel, height, width = data.shape

        tiles_x = width // self.tile_size
        tiles_y = height // self.tile_size

        with torch.no_grad():
            for i in range(tiles_y * tiles_x):
                x = i % tiles_y
                y = math.floor(i/tiles_y)

                ofs_x = y * self.tile_size
                ofs_y = x * self.tile_size

                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                print(input_start_x, input_end_x, input_start_y, input_end_y)

                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y

                input_tile = data[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                output_tile = self.model(input_tile)
                print('{:.2f}%'.format(100 * (i+1) / (tiles_y * tiles_x)))
                
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                output_tile = output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]

                output_tile = output_tile.squeeze(0).transpose(0,2).cpu().numpy().clip(0,1)
                output_tile *= 255
                output_tile = output_tile.astype(np.uint8)
                
                print(output_tile.shape)
                yield output_tile
        
        print('100.00%')

        yield None
    
    def enhance(self, image, output_path=None):
        a = time.time()
        if image.shape[2] == 4:
            image = image[:,:,:3]
        orig_shape = image.shape
        image = cv2.resize(image, (max(orig_shape[1]//self.tile_size * self.tile_size, self.tile_size), max(orig_shape[0]//self.tile_size * self.tile_size, self.tile_size)), interpolation=cv2.INTER_CUBIC)

        print(orig_shape, image.shape)

        tifffile.TiffWriter("enhanced.tif", bigtiff=True).write(
            self.tile_generator(image),
            dtype="uint8",
            shape=(image.shape[0] * 4, image.shape[1] * 4, 3),
            tile=(self.tile_size * 4, self.tile_size * 4),
            compression="lzw",
        )

        im = tifffile.imread('enhanced.tif')
        im = cv2.resize(im, (orig_shape[1]*4, orig_shape[0]*4), interpolation=cv2.INTER_CUBIC)
        if output_path:
            cv2.imwrite(output_path, im)
        #os.remove('enhanced.tif')
        print('Done in {:.2f}s'.format(time.time() - a))
        return im

if __name__ == "__main__":
    upsampler = AstroSleuth(
        scale=4,
        model_path="LATEST",
        blocks=6,
        tile=512,
        tile_pad=32,
        pre_pad=32,
    )

    def upscale(image: Image.Image) -> Image.Image:
        im:np.ndarray = upsampler.enhance(np.array(image))
        im = im[:, :, [2, 1, 0]]
        return Image.fromarray(im)

    main = gr.Interface(
        upscale,
        gr.inputs.Image(type="pil", label="Low Resolution Image"),
        gr.outputs.Image(type="pil", label="Upscaled Image"),
        title="AstroSleuth",
        description="This is an unoptimised demo of AstroSleuth, images will take roughly (W*H)/1,000,000 minutes to process, or roughly 10 minutes on a 4k image.",
        allow_flagging=False,
    )

    main.queue(concurrency_count=5)
    main.launch()
