import math, numpy as np, requests, os, time, warnings, json
from PIL import Image
from importlib import import_module
import torch

class AstroSleuth():
    def __init__(self, tile_size=256, tile_pad=16, wrk_dir="models/", model_name="astrosleuthv2", force_cpu=False, on_download=None, off_download=None):
        # Device selection
        self.device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        
        # Check if model name is known
        model_src:dict = json.load(open("models.json"))["data"]
        assert model_name in model_src, f"Model {model_name} not found! Available models: {list(model_src.keys())}"

        # Load model module
        module_path = model_src[model_name]["src"]["module"]

        self.model_module:torch.nn.Module = getattr(
            import_module(module_path.split("/")[0]),
            module_path.split("/")[1]
        )

        # Download model if not available
        self.model_pth = os.path.join(wrk_dir, f"{model_name}/model.pth")
        self.download(model_src[model_name]["src"]["url"], self.model_pth, model_name, on_download, off_download)
            
        self.wrk_dir = wrk_dir
        self.progress = None

        # Set tile processing parameters
        self.scale = model_src[model_name]["scale"]
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        
    def download(self, src, dst, model_name, on_download=None, off_download=None):
        if not os.path.exists(dst):
            assert not src is None, "That model is not available for downloading - Are you on experimental?"
            os.makedirs(os.path.dirname(dst), exist_ok=True)

            if on_download is not None:
                on_download(model_name)

            with open(dst, 'wb') as f:
                f.write(requests.get(src, allow_redirects=True, headers={"User-Agent":""}).content)

            if off_download is not None:
                off_download()

    def model_inference(self, x: np.ndarray, args:dict={}):
        x = torch.from_numpy(x).to(self.device)
        if not args is None:
            return self.model(x=x, **args).cpu().detach().numpy()
        else:
            return self.model(x=x).cpu().detach().numpy()

    def tile_generator(self, data: np.ndarray, yield_extra_details=False, args={}):
        """
        Process data [height, width, channel] into tiles of size [tile_size, tile_size, channel],
        feed them one by one into the model, then yield the resulting output tiles.
        """

        # [height, width, channel] -> [1, channel, height, width]
        data = np.rollaxis(data, 2, 0)
        data = np.expand_dims(data, axis=0)
        data = np.clip(data, 0, 255)

        batch, channel, height, width = data.shape

        tiles_x = width // self.tile_size
        tiles_y = height // self.tile_size

        for i in range(tiles_y * tiles_x):
            x = i % tiles_y
            y = math.floor(i/tiles_y)

            input_start_x = y * self.tile_size
            input_start_y = x * self.tile_size

            input_end_x = min(input_start_x + self.tile_size, width)
            input_end_y = min(input_start_y + self.tile_size, height)

            input_start_x_pad = max(input_start_x - self.tile_pad, 0)
            input_end_x_pad = min(input_end_x + self.tile_pad, width)
            input_start_y_pad = max(input_start_y - self.tile_pad, 0)
            input_end_y_pad = min(input_end_y + self.tile_pad, height)

            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            input_tile = data[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad].astype(np.float32) / 255

            output_tile = self.model_inference(input_tile, args)
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
    
    def load_model(self):
        model:torch.nn.Module = self.model_module().to(self.device)
        model.load_state_dict(torch.load(self.model_pth, map_location=torch.device(self.device)), strict=False)
        model.eval()
        return model

    def enhance_with_progress(self, image:Image, args:dict={}):    
        """
        Take a PIL image and enhance it with the model, yielding stats about the
        final image and then the final image itself.
        """    

        # Load model only now because when using streamlit, multiple users spawn multiple instances of this class, so 
        # we only load the model when needed. The App() class is responsible for queuing requests to this class
        self.model = self.load_model()
        original_width, original_height = image.size

        # Because tiles may not fit perfectly, we resize to the closest multiple of tile_size
        image = image.resize((max(original_width//self.tile_size * self.tile_size, self.tile_size), max(original_height//self.tile_size * self.tile_size, self.tile_size)), resample=Image.Resampling.BICUBIC)
        image = np.array(image)

        # Initiate a pillow image to save the tiles
        result = Image.new("RGB", (image.shape[1]*self.scale, image.shape[0]*self.scale))
        
        for i, tile in enumerate(self.tile_generator(image, yield_extra_details=True, args=args)):
            
            if tile is None:
                break
            
            tile_data, x, y, w, h, p = tile
            result.paste(Image.fromarray(tile_data), (x*self.scale, y*self.scale))
            yield p
        
        # Resize back to the expected size
        yield result.resize((original_width * self.scale, original_height * self.scale), resample=Image.Resampling.BICUBIC)
        
    def enhance(self, image:Image) -> Image:
        """
        Skips the progress reporting and just returns the final image.
        """
        return list(self.enhance_with_progress(image))[-1]
    
if __name__ == '__main__':
    import sys

    # User ran with only "main.py"
    if not len(sys.argv) == 4: 
        print("Use main.py with a source, destination file, and model, eg: 'python3 main.py img.png upscaled.png astrosleuthv2'")
        print("You might also be interested in using the streamlit interface with: 'streamlit run app.py'")
        quit()

    src = sys.argv[1]
    dst = sys.argv[2]
    model_name = sys.argv[3]

    a = AstroSleuth(model_name=model_name)
    img = Image.open(src)
    r = a.enhance(img)
    r.save(dst)
