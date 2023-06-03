from modules.realesr import Network
import torch

src = "model.pth"

model = Network(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
model.load_state_dict(torch.load(src), strict=True)
model.eval()

x = torch.randn(1, 3, 512, 512)
input_names = ["input"]
output_names = ["output_"]

dynamic_axes_dict = {'input': {0: 'batch_size', 2: 'height', 3: 'width'}, 'output': {0: 'batch_size', 2: 'height', 3: 'width'}}
torch.onnx.export(model, x,  ".".join(src.split(".")[:-1]) + ".onnx", verbose=False, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes_dict, export_params=True)