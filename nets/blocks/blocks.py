import torch
import math
from torch.nn.modules.utils import _triple

class ConvBlock2D(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvBlock2D, self).__init__()
        self.conv_block_2d = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block_2d(x)


class DeConvBlock3D(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(DeConvBlock3D, self).__init__()
        self.deconv_block_3d = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(in_channel, out_channel, kernel_size, stride, padding),
            torch.nn.BatchNorm3d(out_channel),
            torch.nn.ELU()
        )

    def forward(self, x):
        return self.deconv_block_3d(x)


class ConvBlock3D(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvBlock3D, self).__init__()
        self.conv_block_3d = torch.nn.Sequential(
            torch.nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding),
            torch.nn.BatchNorm3d(out_channel),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block_3d(x)

class STConvBlock3D(torch.nn.Module):
    def __init__(self,in_channel, out_channel, kernel_size, stride, padding):
        self.st_conv_block_3d = torch.nn.Sequential(
            STConvBlock(in_channel, out_channel, kernel_size, stride, padding),
            torch.nn.BatchNorm3d(out_channel),
            torch.nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.st_conv_block_3d(x)


class EncoderBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(EncoderBlock, self).__init__()
        self.conv_eb = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        out = self.conv_eb(x)
        return out


class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, scale_factor):
        super(DecoderBlock, self).__init__()
        self.conv_db = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=scale_factor),
            torch.nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1,
                                     padding=1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ConvTranspose2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1,
                                     padding=1),
            torch.nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        out = self.conv_db(x)
        return out


class TSM(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, input, n_frame=4, fold_div=3):
        n_frame = 4
        B, C, H, W = input.shape
        input = input.view(-1, n_frame, H, W, C)
        fold = C // fold_div
        last_fold = C - (fold_div - 1) * fold
        out1, out2, out3 = torch.split(input, [fold, fold, last_fold], -1)

        padding1 = torch.zeros_like(out1)
        padding1 = padding1[:, -1, :, :, :]
        padding1 = torch.unsqueeze(padding1, 1)
        _, out1 = torch.split(out1, [1, n_frame - 1], 1)
        out1 = torch.cat((out1, padding1), 1)

        padding2 = torch.zeros_like(out2)
        padding2 = padding2[:, 0, :, :, :]
        padding2 = torch.unsqueeze(padding2, 1)
        out2, _ = torch.split(out2, [n_frame - 1, 1], 1)
        out2 = torch.cat((padding2, out2), 1)

        out = torch.cat((out1, out2, out3), -1)
        out = out.view([-1, C, H, W])

        return out


class TSM_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,padding):
        super().__init__()
        self.tsm1 = TSM()
        self.t_conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       padding=padding)

    def forward(self, input, n_frame=2, fold_div=3):
        t = self.tsm1(input, n_frame, fold_div)
        t = self.t_conv1(t)
        return t

#https://github.com/ZitongYu/STVEN_rPPGNet/blob/master/models/STVEN.py
class STConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0):
        super(STConvBlock, self).__init__()
        # convert to iterables
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]
        spatial_stride = [1, stride[1], stride[2]]
        spatial_padding = [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride = [stride[0], 1, 1]
        temporal_padding = [padding[0], 0, 0]

        # compute the number of intermediary channels (M) using formula
        # from the paper section 3.5
        intermed_channels = int(math.floor(
            (kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels) / (
                        kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        # the spatial conv is effectively a 2D conv due to the
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = torch.nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                    stride=spatial_stride, padding=spatial_padding)
        self.bn = torch.nn.BatchNorm3d(intermed_channels)
        self.relu = torch.nn.ReLU()   ##   nn.Tanh()   or   nn.ReLU(inplace=True)
        # the temporal conv is effectively a 1D conv, but has batch norm
        # and ReLU added inside the model constructor, not here. This is an
        # intentional design choice, to allow this module to externally act
        # identical to a standard Conv3D, so it can be reused easily in any
        # other codebase
        self.temporal_conv = torch.nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size,
                                    stride=temporal_stride, padding=temporal_padding)

    def forward(self,x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x


class STVEN_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(STVEN_Block, self).__init__()

        self.steven_block = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels,out_channels,kernel_size,stride,padding,bias),
            torch.nn.InstanceNorm3d(out_channels,affine=True,track_running_stats=True),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.steven_block(x)


