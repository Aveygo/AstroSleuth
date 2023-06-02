import numpy as np
import onnxruntime
from PIL import Image
    
class AstroDetect:
    # Detect if an image is a deep space target or not,
    # eg: moon -> False, galaxy -> True, dog -> False, star -> True

    def __init__(self):
        self.net = onnxruntime.InferenceSession("models/astrodetect/astrodetect.onnx", providers=['CPUExecutionProvider'])
        self.img_size = (227, 227)
    
    def is_space(self, img:Image):
        img = img.resize(self.img_size)
        
        img = np.array(img)
        
        # Convert grayscale to RGB, or remove alpha channel        
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)
        else:
            img = img[:,:,:3]

        img = np.rollaxis(img, 2, 0)
        img = img.astype(np.float32) / 255
        img = np.expand_dims(img, axis=0)
        img = np.clip(img, 0, 1)
        
        y = self.net.run(None, {"input": img})[0][0]

        if y[1] > y[0]:
            return True
        else:
            return False

if __name__ == "__main__":
    import time
    a = time.time()
    ad = AstroDetect()
    b = time.time()
    result = ad.is_space("sample.png")
    c = time.time()
    print(f"Init: {b-a:.2f} seconds, Detect: {c-b:.2f} seconds, Success: {result}")