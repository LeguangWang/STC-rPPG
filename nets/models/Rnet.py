import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from ditac import DiTAC
class GCBlock(nn.Module):
    """
    Global Context Block adapted to 3D convolution
    """

    def __init__(self, C, reduction_ratio=16):
        """
        Global Context layer
        :param C: number of input channels
        :param reduction_ratio: reduction ratio
        """
        super(GCBlock, self).__init__()
        self.attention = nn.Conv3d(C, out_channels=1, kernel_size=1)
        self.c12 = nn.Conv3d(C, math.ceil(C / reduction_ratio), kernel_size=1)
        self.c15 = nn.Conv3d(math.ceil(C / reduction_ratio), C, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, block_input):
        N = block_input.size()[0]
        C = block_input.size()[1]
        D = block_input.size()[2]

        attention = self.attention(block_input)
        block_input = nn.functional.softmax(block_input, dim=3)

        block_input_flattened = torch.reshape(block_input, [N, C, D, -1])
        attention = torch.squeeze(attention, dim=3)
        attention_flattened = torch.reshape(attention, [N, D, -1])

        c11 = torch.einsum('bcdf,bdf->bcd', block_input_flattened,
                           attention_flattened)
        c11 = torch.reshape(c11, (N, C, D, 1, 1))
        c12 = self.c12(c11)

        c15 = self.c15(self.relu(torch.layer_norm(c12, c12.size()[1:])))
        cnn = torch.add(block_input, c15)
        return cnn


class CDC_ST(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.6):

        super(CDC_ST, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape

            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2] > 1:
                kernel_diff = self.conv.weight.sum(2).sum(2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
                return out_normal - self.theta * out_diff

            else:
                return out_normal

class Rnet(nn.Module):

    def __init__(self, frames=128):
        super(Rnet, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            DiTAC(cpab_act_type='gelu_cpab', a=-3, b=3,
                               lambda_smooth=5, lambda_smooth_init=3,
                               lambda_var=1, lambda_var_init=2),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            DiTAC(cpab_act_type='gelu_cpab', a=-3, b=3,
                  lambda_smooth=5, lambda_smooth_init=3,
                  lambda_var=1, lambda_var_init=2),
                    )
        self.gc=GCBlock(32)

        self.ConvBlock2 = nn.Sequential(
            CDC_ST(32, 64, [3, 3, 3], stride=1, padding=1, theta=0.6),
            nn.BatchNorm3d(64),
            DiTAC(cpab_act_type='gelu_cpab', a=-3, b=3,
                  lambda_smooth=5, lambda_smooth_init=3,
                  lambda_var=1, lambda_var_init=2),
            CDC_ST(64, 64, [3, 3, 3], stride=1, padding=1, theta=0.6),
            nn.BatchNorm3d(64),
            DiTAC(cpab_act_type='gelu_cpab', a=-3, b=3,
                  lambda_smooth=5, lambda_smooth_init=3,
                  lambda_var=1, lambda_var_init=2),
            nn.MaxPool3d((2, 2, 2), stride=2)

        )
        self.ConvBlock2_1 = nn.Sequential(
            nn.MaxPool3d((2, 2, 2), stride=2),
            CDC_ST(32, 64, 3, stride=1, padding=1)

        )
        self.ConvBlock3 = nn.Sequential(
            CDC_ST(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            DiTAC(cpab_act_type='gelu_cpab', a=-3, b=3,
                  lambda_smooth=5, lambda_smooth_init=3,
                  lambda_var=1, lambda_var_init=2),
            CDC_ST(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            DiTAC(cpab_act_type='gelu_cpab', a=-3, b=3,
                  lambda_smooth=5, lambda_smooth_init=3,
                  lambda_var=1, lambda_var_init=2),
            nn.MaxPool3d((2, 2, 2), stride=2)
        )
        self.ConvBlock3_1 = nn.Sequential(
            nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
            CDC_ST(64, 64, 3, stride=1, padding=1, theta=0.6)
        )

        self.ConvBlock6 = nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1)
        self.ConvBlock7 = nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1)

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4, 1, 1], stride=[2, 1, 1],
                               padding=[1, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4, 1, 1], stride=[2, 1, 1],
                               padding=[1, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ELU(),
            # nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((frames,1,1))
        )
        self.ConvBlock5 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)


    def forward(self, x):
        x_visual = x
        [batch, channel, length, width, height] = x.shape

        x1 = self.ConvBlock1(x)
        x = self.gc(x1)

        x2 = self.ConvBlock2(x) + self.ConvBlock2_1(x)
        x3 = self.ConvBlock3(x2) + self.ConvBlock3_1(x2)

        x5 = self.ConvBlock6(x3)
        x7 = self.ConvBlock7(x5)
        x6 = self.upsample(x7)
        x6 = self.ConvBlock5(x6)

        rPPG = x6.view(-1, length)

        return rPPG,x,x3
        # return rPPG

