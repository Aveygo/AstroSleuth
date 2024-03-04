
"""
Tool to convert pytorch models to onnx / ncnn
You will need binaries "onnx2ncnn" and "ncnnoptimize", which you can build using:

git clone https://github.com/Tencent/ncnn.git
cd ncnn
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_SYSTEM_GLSLANG=ON -DNCNN_BUILD_EXAMPLES=ON -DNCNN_SHARED_LIB=ON ..
make
make install
"""

import sys
sys.path.append('.')

from main import AstroSleuth
import torch, os

src = "AstroSleuthFAST"
astrosleuth = AstroSleuth(model_name = src)
model = astrosleuth.load_model().cpu()

onnx_dst = ".".join(astrosleuth.model_pth.split(".")[:-1]) + ".onnx"
param_dst = ".".join(astrosleuth.model_pth.split(".")[:-1]) + ".param"
bin_dst = ".".join(astrosleuth.model_pth.split(".")[:-1]) + ".bin"

if os.path.exists(onnx_dst): os.remove(onnx_dst)
if os.path.exists(param_dst): os.remove(param_dst)
if os.path.exists(bin_dst): os.remove(bin_dst)

x = torch.randn(1, 3, 512, 512)
input_names = ["data"]
output_names = ["output"]

dynamic_axes_dict = {'data': {0: 'batch_size', 2: 'height', 3: 'width'}, 'output': {0: 'batch_size', 2: 'height', 3: 'width'}}
print("Converting to onnx...")
torch.onnx.export(model, x,  onnx_dst, verbose=False, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes_dict, export_params=True)

#import onnx
#model_onnx = onnx.load(onnx_dst)
#print(onnx.checker.check_model(model_onnx))

print("Done!")
print("Cleaning onnx...")
os.system(f"python3 -m onnxsim {onnx_dst} {onnx_dst}")
print("Done!")
print("Converting to NCNN...")
os.system(f"./utils/onnx2ncnn {onnx_dst} {param_dst} {bin_dst}")
os.system(f"./utils/ncnnoptimize {param_dst} {bin_dst} {param_dst} {bin_dst} 65536")
print("Done!")