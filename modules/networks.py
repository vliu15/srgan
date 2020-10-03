# The MIT License
#
# Copyright (c) 2020 Vincent Liu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    ''' Implements a residual block '''

    def __init__(self, channels: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),

            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.layers(x)


class Generator(nn.Module):
    ''' Implements the SRGAN generator / SRResNet '''

    def __init__(
        self,
        base_channels: int = 64,
        n_res_blocks: int = 16,
        n_ps_blocks: int = 2,
    ):
        super().__init__()

        # input layer
        self.in_layer = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=9, padding=4),
            nn.PReLU(),
        )

        # residual blocks
        res_blocks = []
        for _ in range(n_res_blocks):
            res_blocks += [ResidualBlock(base_channels)]

        res_blocks += [
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
        ]
        self.res_blocks = nn.Sequential(*res_blocks)

        # pixelxhuffle blocks
        ps_blocks = []
        for _ in range(n_ps_blocks):
            ps_blocks += [
                nn.Conv2d(base_channels, 4 * base_channels, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU(),
            ]
        self.ps_blocks = nn.Sequential(*ps_blocks)

        # output layer
        self.out_layer = nn.Sequential(
            nn.Conv2d(base_channels, 3, kernel_size=9, padding=4),
            nn.Tanh(),
        )

    def forward(self, x):
        x_res = self.in_layer(x)
        x = x_res + self.res_blocks(x_res)
        x = self.ps_blocks(x)
        x = self.out_layer(x)
        return x


class Discriminator(nn.Module):
    ''' Implements the SRGAN discriminator '''

    def __init__(
        self,
        base_channels: int = 64,
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, 2 * base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(2 * base_channels, 2 * base_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(2 * base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(2 * base_channels, 4 * base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(4 * base_channels, 4 * base_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(4 * base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(4 * base_channels, 8 * base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(8 * base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(8 * base_channels, 8 * base_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(8 * base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            # replicate nn.Linear with pointwise nn.Conv2d
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(8 * base_channels, 16 * base_channels, kernel_size=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16 * base_channels, 1, kernel_size=1, padding=0),

            # apply sigmoid as F.logsigmoid later for stability
            nn.Flatten(),
        )

    def forward(self, x):
        return self.layers(x)
