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
        x, c = x[0], x[1]

        condition = self.l1(c).view(-1, self.num_feat, 1, 1)
        features = self.block(x + condition) * 0.1
        
        out = self.rdb1(x + features)
        out = self.rdb2(out + features)
        out = self.rdb3(out + features)
        
        return [out * 0.2 + x, c]

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

        self.average = self.bytes2torch(b'RUDlvTWPi70jvj4+buTsvgSBcb+nUd2+PUSAvu11WT7uee29hVKSPszaG78ZOG8+stMTv9Jf2b985BI/Bj+lPSDjoL14Su292lCpvfCNE7+QY8Q+2oi5Pm+AOL/yMCW+6o8MP8hS4zoG+fw92gwtvtTAAb8V+PS9s/r1vROUnL58WYa+fDgjv1Liub701g4/vyMJvSKD5j1dgRg8FjauvoRGqL7k+EK9mccnv8bQsb4FLY4/NAtQvE3dzT1MKQu/DkE4v5QaVT+pJJG+jYlYviRbxT3kZ969I4rIPiD/1jwVvHg/bJ0+vc/IYj+2Ljc9rVS6vOg7yjta5Jg+HMsrPtTBXD91KjE9OCKYvoT0s77n/48+vn/rPvjArD5txlY/TLUGv40/nz579zK9p+2bvg+UGz6SAI28Sll4v549Mb7SHmg+zEf0vjtTEj5QCPI+aT7Zvg7Yib6oFli+21+Cv8GLtj4LC2y+zZtGv8I/ET87Y2S+XL+aPuGBa71JcJg/qzcWvzQr776vfts+TiQNv68/UT07UZk/CegaP+GF/D6wNHU77UUZPWrikD6WP+a+BG4UvrPs076ZNaM9eTwDP86o7j5Z57k7ghmUP+f4gz1Wsmo/7Og4PI7j+D52c22+fMeYPuuN4z5aeLw+qMWHPuhN474BUZg+vApwP1UV9z4gISu7c/d1vgpqFb7g5r2+OD48vF2AHb5FpKO+R//pvow8kT0KZC2+VD6GPxaHTT9+BEK+27buvdqRij4vNC09kcO2vkBVGD9swgc/ejKRv67h870ZIZw+6+7lv+i7370IXKw+tMyNvtGCIr63IQU/Uj32Pq1+Qr4+D+u+9ETTvg4dQz/kRZ4+DzyHPv4bEj8KKJa+zB1RP7JSxTyiaag+nwlGv2AD1Lvqg1I/rEpjPYTUnEB6RNO+uxXUPt3hPj7C0kU/o+fiPaEGkD1aQeo+9MGbvYKbaj6KWS+/Pt+rvnjnnz7gChg/6QesPoNO0D78xSm/jDPfveJ8k77a6dE94Fd6viB3bj9UNSi/xFfbvYOkiT7VrSc+ABKLPkR5dD7N8hY/ZtJkP7bQmD7O8Qi8RhfSPubUXb/UXxK+EfgzPkvRZj0YXAY/YIvmvdIlTj7B4xm/vpWivkOx1D5NTC6/4Wy1P0ntuz1fpJ0+2E0ovxPMxT1mbRc+YhewvtvOaj6mxgA/wpzRPuTgF795wuo9n5ujvtUccD8Q/HC8KG0LP+Y6AT9qqBE+vD8IP8WjDz/Yx9C9wPt7vWJkFD+cnmY+p6eOvVKqw70uD8E+LNQDP+l85j4kx5g+gGZhuybCU76gwEM+jObaPWs/XT3dt9Q+rj0CP2Xvkj6YAmo9mC0Kvv5jhT1WCmC/IMSYvvu3kT3Yi7c+AusvPUkVJj9KHaq/JhbBPhxC8T6yn90+yL6PPko49T64HCq/eIMNPMibUT7IaRe/HopNP+2EQr6vBIu+YV08vppqDT+Cg3a/+1Pqvrws5L+B98Q+3mO0P/RYqb5QHAG+1BipPrqKhj/ErBy/1qDkvUP7Z78xdiQ+DHmdu3QYT76akgq/soHgvaaeVb73OPm+MsmvvjHOrz4z6IU/9c2Gvjp+ar8IloS/PKkYv+b2l76qvGU/1jGhPz+pG7+0hEk+E93KPNNJkb4Pyn6+ShJePlDqCz/3EAe9byUKvwR1lb6rVze/5gwhP3ZuEr8hLFm/dhatPD2ZNj4q/849/uYAP8jG6L6YyxM+b9QMvp1tHj8MFRM/UWVHPrKlaL8S5+O+bDryvhoCH7+qvNA9UF6jvp6XDD5H7Km+KoWrP3/lpr2k6/u+GCJKP9xTaD9hO9G+FMNtvrFTwr1527K++fozv9i33r4cH32/sLazvkDKEb+UtQe/J+SzPtrqir5OrVE/EV4Sv6DCAzrWsge/edU2vir/ZL077Q/ASrQDvziTtz4/3jw+4blZvwUaer0Y6YE+ZpX7PirQUb5r4DY/s7ZWP6r1QT2PhgO/KgJ4PuCVSD4FSfk+8kz9Ppz8IL+kWAc/liFCPag9oz5EsNg+ED4qPliFBT+gbz0/93jDP5AUQr2q3+K+NuoiPQiUmj5V47i9IvmlPotFjD7T4DM/55uavsykxT7t/D69sQGsPkpZSD4HllW9i6VZPUyB7T2+JY++RsJcPhAP5j6seKY+T83MPaOSgD7Ybye/GrFUPpLG3D7WqbG/DbA0vOIO9T28Ba68vujPPl4a6L4H78e+VvMjP491Rb7yBjE/WZRgvtBgSrx0il4+oorYPi7imbw6n5q+2+M0PUbAnr8/D5y/XAwZvw1HxL5SCqA+X6fVPXAZr7uAlwS/jzytv+2YsT2ndDm/p3xmvmJB8b6woug/+ZTevpeeYb9mqDw9WLusO21mX79Im0Q/nvXHvuzxpb34+UI/TKgKP+/Qxr6pY20/HFlOvsMw4j6xutA9jnz4PfJlz74Aka28xB7CPgIXyT3Jsaa+WrzZPiEHi76SRyi/39MkPwZ7mD0oH/A+iNLXvlJTdD7ikjy/fj50vqqhur6Fjxg/P9TNvrhBZ77Lu0k+ZNgrvl9yHr//1cw/VnKrvh9uuD7JX0g+ZG6HvgrMpzyEQ0C+10G2vlJj8b44p/E+Hcukvihm9z1ukpy+h8YEvjBS5r5Q3Mm+xDKgPfHztz3MP3m8WMY4vRTfyz4yTs89sh6RvkEKcL5OaIC+z6OiPuyY3L6iuow+0y1Uv3+i7z4=')
        self.stars = self.bytes2torch(b'qfQjPw87X79Ewlg+bBKHvlRc+L6bv5W/zZWgPj3g9byRO+6+Sn39vtOBPD9IM7M+j2GLPntbir+iJHA+k6udv3iOzL5+xUS/44IcPgtC6bzdy72+gj+JPhMR9j5EvaQ/KfuaPk/0mj4w7JC+sKwnvgatlz4/npc/FvUVPv1gWz7fQ4K/VVHuPkmBhz8Krqo+U8yTPo0wZT6W2O6+DxoVv5KMEz4UczG+aUnHv3ZNIr8XGnk/0y4yvghqFT/GbR2+IEhUv6soaz9VWNu+2xsSv8YAR7/ENuE9RLacP0A/Gj6pXQM/RwHWvjVSQD/MBn497LDJvnfWGr8eGfi9khlCPkKN6T4PxPO+bFS/vrFZ1b4hxhE/Fq5Yvs44n75Hc4A/NNA9vtT3Rz4IkRM+h88hvyGk7T3e06I+IH9Pv2P7ZD6w3cc9yz/XPt2WAj9cH+K9tcvaPbdv4j599+++Kq4IO2MDE72ce2g+lX+av9cWFj+e2+q+GwrnPj6Bdr6hjtI+STy0v38NOr0wYqQ+UAeWvzOhMj4CyTs/6GsJP+pVCj8OugA/YfA2Pl1Rpj5J7dG+AkYDv8xYYb6SIkG/lElwvhkso77IBdo+hGgTPiiFbj6y4xi+5NWPv9PNkT+5Xai+X5ckP5PH8j7HldM+GzJnPu96i7/uUbQ+g1L2Pj2OIr9Ms0K/BYkAv0ffEr9Tl5k/n67rPGzlQT/c1xG+2JYxP4yHOL+31Qm+WgziPSxaEr+sCcc+uTGkP+5Jxj4jyIK/VtsAP3lPGT6lV909fUUXv54nMT9EzIy+uHQAwP5yPr+sKL8+j7LuPQiS5L4gSju+A6OdP6SsLj+4zbC9GoQ3vyKk5b4tcEC/BCn9viaOib+p+ky/gNz5Pqy/XL7Dqzc/5CjoPoXPwD1IN4M/DDuxPTKW1z/02oS/vK+OPocIHD6VOoc+j0gvv4EUVT0aqRI/UTxfP52/fT8gSnO+0QDEPZ/ylj+Y+Dc/0fMzPy+jCL8TXkG/F+WAPaINEb8wNos7d1fAvgxWUj4Hzys+k5twv72Gdb1BPea9XnczPip8JL+6Iiw+6jmvPlufaz+Vc6E+3KmIPsboU78ZOjy/fGEyv3eiw77hGkQ/SnLYPhmWbD6Kf1K+HUuBPmVg0L2ZDuy+ObijQHMIgD/YpZu+JOfpPNVfGr42OEu81w2tvpSSKb7Q+Qs/Jpx1PuATaT8xn+A9mpgUPkrf974jcUs/ANz6vqdAuz8RDjG+lV6yPTSnqD90wu2+dtrwPkkbZL8IqC0+ari0PvSZoj8E1fA+MYIZP55xUz+/0I8+OWRQP0x9Lj67NpO8xxl4vqHnBD8qS6s+D8P/PofEKj5rCX6+gu5TPjiqjr4OGr28/HfVPU6WIz9leOs+dRqiPgWCnz9sGLTAs7iqvi2pSD38P98+cm+PP7eAz7rI8gK+ELVbv4xTjr5nONK+QT9dvnJZrb6I8LO+p0tYvrFcgj41r6q/VWSZvvzno79H05k95rMHPQyEYb+upTY/5SfEPlbiFz8cAvO+72amvVx3P7+zjQC/gv7SPh1jYr+kbfI+/1GLPx/X8T2lbH6+gABDP2nnjj/nu7C/6HhfP4S2nL/EVhU/q8+iv0jwAb+dK889EOYvPmK1Xz7RwL694UnJPIzZAb+TcTI+9Lu9vvD+Dz4ETcK+WJ1NvyIhWL2f3Y8+XFXqvgxFiT14VgK/tR09v1E63D4TPjk+HcHyvqLBCr+8FrM/gjiLP3l+jj/HyWY+h8kJP+iJyb4EODi+lBviviC647/QVeG8PzYJv7DnCr0QwRw/WETsvGR7tb5jgTm/UE2NPlohiT9xkl49A+ttvpw1Qr7yBZo9U6STvwZcMj4Um94+GcKFP81tUL8dd5098T6PPpIz072BPtU8kZ3nPgdLTr6+pdu+lgVqPjuBBL9g/ae+VZ31PtWKAr6fLW2/giDyvnAo773QRR4/7WjwPppmKj1kIQW/98UtPqjM+70CEsq9pSVLv5KDAr6xD9a+AA7cPkSUDL8M49S+JlQBv8O2r77fJBI/3LIzv1Vfk7wyhhQ+yG/wPYC05r/ryNC9L26ZP5yc3T5b3RC/sVRbPuiyiL6iyak/OOJTPmUwGr5LHAC/eNOXP4P8GD9hPIA/wOXZvjWCALwpHb0+nhFfPuuMBT7+IIw+QJ63Pur25b7gKxq/axc+vh0ufb5Yt3K+1+DmvgRUqL5DDhm/Wi0iv4eMPL7mcDC/gmTWvvjuaL7DjiU/M9k5vz54yL96CVk9ip8Fvo2C2T6hB5e/OimDPxwLlzo4HkW+yiLiPjghfr8RPbI+SzUhPzG0ET7IyTy/HOanv2baNj/3DgC/8773PofZ7L6tVF0/IPEEvp4x5r41u04+Yz/KPMQ0f78bFIg+Zop9PltNOD9hVoA/SZfGvge0070QzZo/x/OJPo6pDL5iCfA9xrM1PhXgxjwr2zE/dxfyvkHyh75DzG+9e3gmP1zXOD59e50+XJ2LPW81Kr/AwAc/EclrvzfwFz8aoQo/dHMwvsc1Yb1JW6s+tN6Bv230nz3CUnO/WXY0vwX5ej8HGEc/LztUvi+/9r0mlM0+OOOsvQiWv75sdxS/sBvGvnpzj79/nv4+ohS1vSXDnr4QrUY/TJIevxBJNT5vLL8+iZ0qP0ncWL7Bbw+97OWcPiMAbT3dOoi+wNnrvVPzpr9nfLS+YbhNPj3dDL/yRoM/Ix1rv4/wGD8=')
        self.detail = self.bytes2torch(b'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')
                                      
    def bytes2torch(self, x:bytes):
        x = np.frombuffer(base64.decodebytes(x), dtype=np.float32).copy()
        return torch.from_numpy(x)

    def forward(self, x, star_strength, detail_strength):
        c = self.average + self.stars * star_strength + self.detail * detail_strength
        c = c.to(x.device).float()
        c = self.lrelu(self.l1(c))
        
        x = self.conv_first(x)

        skip = self.conv_body(self.body([x, c])[0])
        x = x + skip
        
        x = self.lrelu(self.conv_up1(F.interpolate(x, scale_factor=2, mode='bilinear')))
        x = self.lrelu(self.conv_up2(F.interpolate(x, scale_factor=2, mode='bilinear')))

        out = self.conv_last(self.lrelu(self.conv_hr(x)))
        return out
