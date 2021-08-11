import torch

from nets.blocks.blocks import STVEN_Block, STConvBlock
from ..funcs.complexFunctions import complex_matmul


class DAModule(torch.nn.Module):
    def __init__(self, in_channels):
        super(DAModule, self).__init__()
        self.inter_channels = in_channels // 4
        self.conv_p1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.inter_channels, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(self.inter_channels),
            torch.nn.ReLU(True)
        )
        self.conv_c1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.inter_channels, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(self.inter_channels),
            torch.nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(self.inter_channels)
        self.cam = _ChannelAttentionModule()
        self.conv_p2 = torch.nn.Sequential(
            torch.nn.Conv2d(self.inter_channels, self.inter_channels, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(self.inter_channels),
            torch.nn.ReLU(True)
        )
        self.conv_c2 = torch.nn.Sequential(
            torch.nn.Conv2d(self.inter_channels, self.inter_channels, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(self.inter_channels),
            torch.nn.ReLU(True)
        )

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c
        return feat_fusion


class _PositionAttentionModule(torch.nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = torch.nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = torch.nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = torch.nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = torch.nn.Parameter(torch.zeros(1))
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ChannelAttentionModule(torch.nn.Module):
    """Channel attention module"""

    def __init__(self):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = torch.nn.Parameter(torch.zeros(1))
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


class ChannelAttentionModule(torch.nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = torch.nn.Sequential(
            torch.nn.Conv2d(channel, channel // ratio, 1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        # print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(torch.nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(torch.nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = complex_matmul(self.channel_attention(x), x)
        # print('outchannels:{}'.format(out.shape))
        out = complex_matmul(self.spatial_attention(out), out)
        return out


class ResBlock_CBAM(torch.nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(ResBlock_CBAM, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(places),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1,
                            bias=False),
            torch.nn.BatchNorm2d(places),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1,
                            bias=False),
            torch.nn.BatchNorm2d(places * self.expansion),
        )
        self.cbam = CBAM(channel=places * self.expansion)

        if self.downsampling:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1,
                                stride=stride, bias=False),
                torch.nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        # print(x.shape)
        out = self.cbam(out)
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class STVENDownSamplingModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, label_channels):
        super(STVENDownSamplingModule, self).__init__()

        self.stven_block1 = STVEN_Block(in_channels + label_channels, out_channels, (3, 7, 7), (1, 1, 1), (1, 3, 3))
        self.stven_block2 = STVEN_Block(out_channels, out_channels * 2, (3, 4, 4), (1, 2, 2), (1, 1, 1))
        self.stven_block3 = STVEN_Block(out_channels * 2, out_channels * 8, (4, 4, 4), (2, 2, 2), (1, 1, 1))

    def forward(self, x):
        x = self.stven_block1(x)
        x = self.stven_block2(x)
        x = self.stven_block3(x)
        return x


class STVENBottleneck(torch.nn.Module):
    def __init__(self, in_channels, repeat):
        super(STVENBottleneck, self).__init__()
        self.layers = []
        for i in range(repeat):
            self.layers.append(STConvBlock(in_channels, in_channels, [3, 3, 3], (1, 1, 1), [1, 1, 1]))

    def forward(self, x):
        return self.layers(x)


class STVENUpSamplingModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(STVENUpSamplingModule, self).__init__()

        self.stven_block1 = STVEN_Block(in_channels, in_channels // 4, (4, 4, 4), (2, 2, 2), (1, 1, 1))
        self.stven_block2 = STVEN_Block(in_channels // 4, in_channels // 8, (1, 4, 4), (1, 2, 2), (0, 1, 1))
        self.out_conv3d = torch.nn.Conv3d(in_channels // 16, out_channels, (1, 7, 7), (1, 1, 1), (0, 3, 3), False)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.stven_block1(x)
        x = self.stven_block2(x)
        x = self.out_conv3d(x)
        x = self.tanh(x)
        return x


class MixAttentionModule(torch.nn.Module):
    '''
    Spatial-skin attention module
    '''

    def __init__(self):
        super(MixAttentionModule, self).__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.AVGpool = torch.nn.AdaptiveAvgPool1d(1)
        self.MAXpool = torch.nn.AdaptiveMaxPool1d(1)

    def forward(self, x, skin):
        """
            inputs :
                x : input feature maps( B X C X T x W X H)
                skin : skin confidence maps( B X T x W X H)
            returns :
                out : attention value
                spatial attention: W x H
        """
        m_batchsize, C, T, W, H = x.size()
        B_C_TWH = x.view(m_batchsize, C, -1)
        B_TWH_C = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        B_TWH_C_AVG = torch.sigmoid(self.AVGpool(B_TWH_C)).view(m_batchsize, T, W, H)
        B_TWH_C_MAX = torch.sigmoid(self.MAXpool(B_TWH_C)).view(m_batchsize, T, W, H)
        B_TWH_C_Fusion = B_TWH_C_AVG + B_TWH_C_MAX + skin
        Attention_weight = self.softmax(B_TWH_C_Fusion.view(m_batchsize, T, -1))
        Attention_weight = Attention_weight.view(m_batchsize, T, W, H)
        # mask1 mul
        output = x.clone()
        for i in range(C):
            output[:, i, :, :, :] = output[:, i, :, :, :].clone() * Attention_weight

        return output, Attention_weight
