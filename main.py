import math, numpy as np, requests, os, time
from PIL import Image
import onnxruntime
from astrodetect import AstroDetect

class AstroSleuth():
    def __init__(self, tile_size=256, tile_pad=16, src="models/astrosleuth/model.onnx", use_detector=True):
        self.tile_size = tile_size
        self.tile_pad = tile_pad

        self.download("https://t.ly/fJ3D", src) # AstroSleuthV1
        #self.download("https://t.ly/2SAQ", src) # AstroSleuthV2

        self.detector = None
        if use_detector:
            self.download("https://t.ly/NiNM", "models/astrodetect/astrodetect.onnx")
            self.download("https://t.ly/RZ_Y", "models/realesr/realesr.onnx")
            self.detector:AstroDetect = AstroDetect()

        self.src = src
        self.scale = 4
        self.progress = None
    
    def download(self, src, dst):
        if not os.path.exists(dst):
            with open(dst, 'wb') as f:
                f.write(requests.get(src, allow_redirects=True, headers={"User-Agent":""}).content)

    def model(self, x: np.ndarray):
        return self.onnx_model.run([self.output_name], {self.input_name: x})[0]

    def tile_generator(self, data: np.ndarray, yield_extra_details=False):
        data = np.rollaxis(data, 2, 0)
        data = np.expand_dims(data, axis=0)
        data = np.clip(data, 0, 255)

        batch, channel, height, width = data.shape

        tiles_x = width // self.tile_size
        tiles_y = height // self.tile_size

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

            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            input_tile = data[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad].astype(np.float32) / 255

            output_tile = self.model(input_tile)
            self.progress = (i+1) / (tiles_y * tiles_x)
            
            output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
            output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
            output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

            output_tile = output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]

            output_tile = (np.rollaxis(output_tile, 1, 4).squeeze(0).clip(0,1) * 255).astype(np.uint8)
            
            if yield_extra_details:
                yield (output_tile, input_start_x, input_start_y, input_tile_width, input_tile_height, self.progress)
            else:
                yield output_tile
        
        yield None

    def enhance_with_progress(self, image:Image) -> Image:

        model_src = self.src
        if self.detector is not None:
            a = time.time()
            result = self.detector.is_space(image)
            print(f"Detection took {time.time() - a:.2f} seconds, result: {result}")
            if not result:
                model_src = "models/realesr/realesr.onnx"

        self.onnx_model = onnxruntime.InferenceSession(model_src, providers=["CUDAExecutionProvider"])
        self.input_name = self.onnx_model.get_inputs()[0].name
        self.output_name = self.onnx_model.get_outputs()[0].name

        original_width, original_height = image.size
        image = image.resize((max(original_width//self.tile_size * self.tile_size, self.tile_size), max(original_height//self.tile_size * self.tile_size, self.tile_size)), resample=Image.Resampling.BICUBIC)
        image = np.array(image)[:,:,:3]

        result = Image.new("RGB", (image.shape[1]*self.scale, image.shape[0]*self.scale))
        for i, tile in enumerate(self.tile_generator(image, yield_extra_details=True)):
            if tile is None: break
            tile_data, x, y, w, h, p = tile
            result.paste(Image.fromarray(tile_data), (x*self.scale, y*self.scale))
            yield p
        result = result.resize((original_width * self.scale, original_height * self.scale), resample=Image.Resampling.BICUBIC)
        yield result
        
    def enhance(self, image:Image) -> Image:
        return list(self.enhance_with_progress(image))[-1]
    
if __name__ == '__main__':
    import sys
    src = sys.argv[1]
    dst = sys.argv[2]
    a = AstroSleuth()
    r = a.enhance(Image.open(src))
    r.save(dst)