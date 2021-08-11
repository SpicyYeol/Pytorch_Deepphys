# https://github.com/ZitongYu/STVEN_rPPGNet/blob/master/models/STVEN.py
import torch
from nets.modules.modules import STVENDownSamplingModule,STVENBottleneck,STVENUpSamplingModule

class STVENGeneratorModel(torch.nn.Module):
    def __init__(self,in_channels,out_channels,label_channels,repeat = 4):
        super(STVENGeneratorModel, self).__init__()
        self.down_sampling_module = STVENDownSamplingModule(in_channels,out_channels,label_channels)
        self.bottleneck = STVENBottleneck(out_channels*8,repeat)
        self.up_sampling_module = STVENUpSamplingModule(out_channels*8,out_channels)

    def forward(self,x,c):
        '''
        :param x: x = input data
        TODO: FIND what is c
        :param c: c =
        :return:
        '''
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1, 1)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3), x.size(4))

        x0 = torch.cat([x,c], dim = 1)
        x0 = self.down_sampling_module(x0)
        x0 = self.bottleneck(x0)
        x0 = self.up_sampling_module(x0)

        return x0 + x

