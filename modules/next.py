from torch.nn import functional as F
from torch import nn as nn
import torch, base64, numpy as np

import streamlit as st

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            # layernorm not implemented in ncnn... 
            # return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, unbiased=False, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = x * self.weight + self.bias
            return x
        
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
    
class Block(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return x

class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.num_feat = num_feat

        # NOTE: Does not play nice with NCNN - maybe GRN norm?
        self.block = Block(num_feat)
        self.l1 = nn.Linear(num_feat, num_feat)

        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        x, c, s = x[0], x[1], x[2]

        if not c is None:
            condition = self.l1(c).view(-1, self.num_feat, 1, 1)
            features = self.block(x + condition) * 0.1 * s
        else:
            features = torch.zeros_like(x)
        
        out = self.rdb1(x + features)
        out = self.rdb2(out + features)
        out = self.rdb3(out + features)
        
        return [out * 0.2 + x, c, s]

def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

def pixel_unshuffle(x, scale):
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)

def streamlit():
    detail_strength = 0.0
    star_strength = 0.0
    cond_strength = 0.5
    spikes_strength = 0

    col1, col2 = st.columns([1,1])
    with col1:
        color_matching = st.toggle("Use color matching", help="Force the colors to match the input, not good for noisy inputs")
    with col2:
        use_cond = st.toggle("Use conditioning", help="Guide the model into specific styles")
    
    

    col1, col2, col3 = st.columns([1, 1, 1])
    if use_cond:

        cond_strength = st.slider("Strength", 0.0, 5.0, 0.5, 0.1, help="How strong should the guiding be. Recommended less than 1 for suble effects.")

        with col1:
            spikes_strength = st.slider("Spikes", -1.0, 1.0, 0.0, 0.1, help="Controls the amount of diffraction spikes")
        with col2:
            detail_strength = st.slider("Detail", -1.0, 1.0, 0.0, 0.1, help="Controls the amount of fine details, may introduce noise if too high.")
        with col3:
            star_strength = st.slider("Stars", -1.0, 1.0, 0.0, 0.1, help="Controls the amount of additional stars in the output.")
        
    return {
        "detail_strength": detail_strength, 
        "star_strength": star_strength, 
        "spikes_strength": spikes_strength,
        "cond_strength": cond_strength,
        "use_cond":use_cond,
        "color_matching": color_matching
    }

class Network(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=6, num_grow_ch=32):
        super(Network, self).__init__()
        self.scale = scale
        self.num_feat = num_feat

        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.l1 = nn.Linear(512, num_feat)

        #self.average = self.bytes2torch(b'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')
        self.average = self.bytes2torch(b'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')
        #self.detail = self.bytes2torch(b'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')
        self.detail = self.bytes2torch(b'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')
        self.stars = self.bytes2torch(b'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')
        self.spikes = self.bytes2torch(b'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')
        
    def bytes2torch(self, x:bytes):
        x = np.frombuffer(base64.decodebytes(x), dtype=np.float32).copy()
        return torch.from_numpy(x)

    def rgb_to_ycbcr(self, image: torch.Tensor) -> torch.Tensor:
        r: torch.Tensor = image[..., 0, :, :]
        g: torch.Tensor = image[..., 1, :, :]
        b: torch.Tensor = image[..., 2, :, :]
        y: torch.Tensor = .299 * r + .587 * g + .114 * b
        cb: torch.Tensor = (b - y) * .564 + .5
        cr: torch.Tensor = (r - y) * .713 + .5
        return y, cb, cr
    
    def ycbcr_to_rgb(self, y: torch.Tensor, cb: torch.Tensor, cr: torch.Tensor) -> torch.Tensor:
        r: torch.Tensor = y + 1.403 * (cr - 0.5)
        g: torch.Tensor = y - 0.714 * (cr - 0.5) - 0.344 * (cb - 0.5)
        b: torch.Tensor = y + 1.773 * (cb - 0.5)
        rgb_image: torch.Tensor = torch.stack([r, g, b], dim=-3)
        return torch.clamp(rgb_image, 0, 1)

    def color_matching(self, src, trg, match_strength=1):
        """
        Match the colors of src onto trg
        We assume that the luminance of trg provides enough detail.
        However, if the src was noisy or already low quality (eg, jpeg), then it may be
        better to decrease the match_strength
        """
        src_y, src_cb, src_cr = self.rgb_to_ycbcr(src)
        trg_y, trg_cb, trg_cr = self.rgb_to_ycbcr(trg)
        src_cb = F.interpolate(src_cb.view(1, 1, src.shape[2], src.shape[3]), scale_factor=4, mode='bilinear').view(1, src.shape[2]*4, src.shape[3]*4)
        src_cr = F.interpolate(src_cr.view(1, 1, src.shape[2], src.shape[3]), scale_factor=4, mode='bilinear').view(1, src.shape[2]*4, src.shape[3]*4)
        trg_cb = trg_cb * (1-match_strength) + src_cb * match_strength
        trg_cr = trg_cr * (1-match_strength) + src_cr * match_strength
        trg = self.ycbcr_to_rgb(trg_y, trg_cb, trg_cr)
        return trg

    def forward(self, x, star_strength=0, detail_strength=0, spikes_strength=0, use_cond=False, cond_strength=1, color_matching=False):
        c = self.average + self.stars * star_strength + self.detail * detail_strength + self.spikes * spikes_strength #  self.average +
        c = self.lrelu(self.l1(c.to(x.device).float())) if use_cond else None
    
        x_primary = self.conv_first(x)

        skip = self.conv_body(self.body([x_primary, c, cond_strength])[0])
        x_primary = x_primary + skip
        
        x_primary = self.lrelu(self.conv_up1(F.interpolate(x_primary, scale_factor=2, mode='bilinear')))
        x_primary = self.lrelu(self.conv_up2(F.interpolate(x_primary, scale_factor=2, mode='bilinear')))

        out = self.conv_last(self.lrelu(self.conv_hr(x_primary)))
        if color_matching:
            return self.color_matching(x, out)
        else:
            return out

if __name__ == "__main__":
    import json
    network = Network()

    data = {}
    data["average"] = network.average.tolist()
    data["detail"] = network.detail.tolist()
    data["spikes"] = network.spikes.tolist()
    data["stars"] = network.stars.tolist()

    with open('next_conds.json', 'w') as f:
        json.dump(data, f)