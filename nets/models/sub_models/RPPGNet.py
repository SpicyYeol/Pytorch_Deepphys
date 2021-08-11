import torch
import math

from nets.blocks.blocks import STConvBlock3D, ConvBlock3D
from nets.modules.modules import MixAttentionModule

class RPPGNet(torch.nn.Module):
    def __init__(self, frames = 64,  repeat = 4):
        super(RPPGNet, self).__init__()
        self.repeat = repeat
        self.ConvSpa = torch.nn.Sequential(                             #     MAIN BRANCH      Input            Comment
            ConvBlock3D(3, 16, [1, 5, 5], [1, 1, 1], [0, 2, 2]),        #01 ConvSpa1     : [3, 64, 128, 128]
            torch.nn.AvgPool3d((1, 2, 2), (1, 2, 2)),                   #02 AvgPoolSpa   : [16, 64, 64,  64]
            STConvBlock3D(16, 32, [3, 3, 3], [1, 1, 1], [1, 1, 1]),     #03 ConvSpa3     : [32, 64, 64,  64]
            STConvBlock3D(32, 32, [3, 3, 3], [1, 1, 1], [1, 1, 1]),     #04 ConvSpa4     : [32, 64, 64,  64]    x_visual6464 use for skin segmentation branch
            torch.nn.AvgPool3d((1, 2, 2), (1, 2, 2)),                   #05 AvgPoolSpa   : [32, 64, 32,  32]
            STConvBlock3D(32, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]),     #06 ConvSpa5     : [64, 64, 32,  32]
            STConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]),     #07 ConvSpa6     : [64, 64, 32,  32]    x_visual3232 use for skin A1 loss
            torch.nn.AvgPool3d((1, 2, 2), (1, 2, 2)),                   #08 AvgPoolSpa   : [64, 64, 16,  16]
            STConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]),     #09 ConvSpa7     : [64, 64, 16,  16]
            STConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]),     #10 ConvSpa8     : [64, 64, 16,  16]
            STConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]),     #11 ConvSpa9     : [64, 64, 16,  16]
            torch.nn.Conv3d(64, 1, [1, 1, 1], [1, 1, 1], [0, 0, 0]),    #12 ConvSpa10    : [64, 64,  1,   1]
            torch.nn.Conv3d(64, 1, [1, 1, 1], [1, 1, 1], [0, 0, 0]),    #13 ConvSpa11    : [64, 64,  1 ,  1]
        )

        self.skin_main = torch.nn.Sequential(
            ConvBlock3D(32, 16, [1, 3, 3], [1, 1, 1], [0, 1, 1]),
            ConvBlock3D(16,  8, [1, 3, 3], [1, 1, 1], [0, 1, 1])
        )

        self.skin_residual = torch.nn.Sequential(
            ConvBlock3D(32, 8, [1, 1, 1], [1, 1, 1], [0, 0, 0])
        )

        self.skin_output = torch.nn.Sequential(
            torch.nn.Conv3d( 8 ,1 , [1,3,3], [1,1,1], [0,1,1]),
            torch.nn.Sigmoid()
        )

        self.ConvPart = [torch.nn.Conv3d(frames,1,[1,1,1], [1,1,1], [0,0,0] ) for i in range(repeat)]

        self.avg_poolSkin_down = torch.nn.AvgPool2d((2, 2), stride=2)
        self.poolSpa = torch.nn.AdaptiveMaxPool3d((frames,1,1))
        self.MixA_Module = MixAttentionModule()

    def forward(self, x):
        x_visual = x # need to residual computation
        modules = list(self.ConvSpa.modules())
        # TODO : need to change efficiently
        # main branch
        for m in modules[:4]:
            x = m(x)
        x_visual6464 = x
        for m in modules[4:7]:
            x = m(x)
        x_visual3232 = x
        for m in modules[7:11]:
            x = m(x)
        x_visual1616 = x

        # branch 1 : skin segmentation
        x_skin_main = self.skin_main(x_visual6464)
        x_skin_residual = self.skin_residual(x_visual6464)
        x_skin = self.skin_output(x_skin_main+x_skin_residual)
        x_skin = x_skin[:, 0, :, :] # [74,64,64]

        # SkinA1_loss
        x_skin3232 = self.avg_poolSkin_down(x_skin)
        x_visual3232_SA1, attention3232 = self.MixA_Module(x_visual3232, x_skin3232)
        x_visual3232_SA1 = self.poolSpa(x_visual3232_SA1)
        ecg_SA1 = modules[-2](x_visual3232_SA1).squeeze(1).squeeze(-1).squeeze(-1)

        # SkinA2_loss
        x_skin1616 = self.avg_poolSkin_down(x_skin)
        x_visual1616_SA2, attention1616 = self.MixA_Module(x_visual1616, x_skin1616)
        gloabl_F = self.poolSpa(x_visual1616_SA2)
        ecg_global = modules[-1](gloabl_F).squeeze(1).squeeze(-1).squeeze(-1)

        # Local ecg loss
        length = math.sqrt(self.repeat)
        '''
        ecg part(i,j) = ecg part ( i* length + j)
        ┌───────┬───────┐
        │ (0,0) │ (0,1) │
        ├───────┼───────┤
        │ (1,0) │ (1,1) │
        └───────┴───────┘
        '''
        ecg_part = [self.ConvPart[i*length+j](self.poolSpa(x_visual1616_SA2[:,:,:,8*j : 8*(j+1), 8*i : 8*(i+1)])) for i in range(length) for j in range(length)]

        return x_skin, ecg_SA1, ecg_part, x_visual6464, x_visual3232