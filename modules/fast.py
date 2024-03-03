from torch import nn as nn
from torch.nn import functional as F

class SRVGGModified(nn.Module):
    # https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/archs/srvgg_arch.py
    
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super(SRVGGModified, self).__init__()
        self.upscale = upscale

        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))

        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))

        self.conv_up1 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(32, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.lrelu(self.conv_up1(F.interpolate(out, scale_factor=2, mode='bilinear')))
        out = self.conv_up2(F.interpolate(out, scale_factor=2, mode='bilinear'))

        return out + F.interpolate(x, scale_factor=self.upscale, mode='bilinear')