import torch
from safetensors import safe_open
from safetensors.torch import save_file
from modules.realesr import Network 

model = torch.load("models/AstroSleuthV1/model.pth", map_location=torch.device('cpu'))
save_file(model, "models/AstroSleuthV1/model.safetensors")