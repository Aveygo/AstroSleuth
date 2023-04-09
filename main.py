from app import AstroSleuth
import numpy as np
from PIL import Image
import sys

if __name__ == '__main__':
    image = np.asarray(Image.open(sys.argv[1]))
    upsampler = AstroSleuth(
        scale=4,
        model_path="models/generator.pth",
        blocks=6,
        tile=512,
        tile_pad=32,
        pre_pad=32,
    )
    
    output = upsampler.enhance(image, sys.argv[2])