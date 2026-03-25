import torch
import torch.nn as nn

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    ),
    'se_block': lambda C, stride, affine: SEBlock(C, C, kernel_size=1, stride=stride, affine=affine),
}


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # GAP
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # GMP

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        x_cat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


# CBAM
# class SepConv(nn.Module):
#     def __init__(self, C_in, C_out, kernel_size, stride, padding,
#                  affine=True, use_cbam=True, cbam_ratio=16, cbam_kernel=7):
#         super(SepConv, self).__init__()
#         self.op = nn.Sequential(
#             nn.ReLU(inplace=False),
#             nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
#                       groups=C_in, bias=False),
#             nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(C_in, affine=affine),
#             nn.ReLU(inplace=False),
#             nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding,
#                       groups=C_in, bias=False),
#             nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(C_out, affine=affine),
#         )
#
#         self.use_cbam = use_cbam
#         if use_cbam:
#             self.cbam = CBAM(C_out, ratio=cbam_ratio, kernel_size=cbam_kernel)
#
#     def forward(self, x):
#         out = self.op(x)
#         if self.use_cbam:
#             out = self.cbam(out)
#         return out


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


# CBAM
# class DilConv(nn.Module):
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation,
#                  affine=True, use_cbam=True, cbam_ratio=16, cbam_kernel=7):
#         super(DilConv, self).__init__()
#         self.op = nn.Sequential(
#             nn.ReLU(inplace=False),
#             nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
#                       dilation=dilation, groups=C_in, bias=False),
#             nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(C_out, affine=affine),
#         )
#
#         self.use_cbam = use_cbam
#         if use_cbam:
#             self.cbam = CBAM(C_out, ratio=cbam_ratio, kernel_size=cbam_kernel)
#
#     def forward(self, x):
#         out = self.op(x)
#         if self.use_cbam:
#             out = self.cbam(out)
#         return out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for DARTS op space"""

    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=0,
                 reduction=16, affine=True):
        super(SEBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

        # SE 注意力分支
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(C_out, C_out // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(C_out // reduction, C_out, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv(x)  # 卷积特征提取
        b, c, _, _ = out.size()
        y = self.global_pool(out).view(b, c)  # Squeeze
        y = self.fc(y).view(b, c, 1, 1)  # Excitation
        out = out * y.expand_as(out)  # 通道加权
        return out


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out
