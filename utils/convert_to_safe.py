import torch
from safetensors import safe_open
from safetensors.torch import save_file
from modules.realesr import Network 

model = torch.load("models/AstroSleuthNEXT/model.pth", map_location=torch.device('cpu'))
print(model.keys())
#save_file(model, "models/AstroSleuthNEXT/model.safetensors")