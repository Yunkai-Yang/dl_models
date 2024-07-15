import torch
import torch.nn as nn


class StochasticDropout(nn.Module):
    def __init__(self, survival_prob):
        super().__init__()
        self.survival_prob = survival_prob

    def forward(self, x, training):
        if training:
            drop_prob = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
            x = torch.div(x, self.survival_prob) * drop_prob
        return x


def getActFunc(act_func):
    if act_func is None:
        act_func = nn.Identity
    elif act_func == 'default':
        act_func = nn.SiLU
    else:
        getattr(nn, act_func)
    return act_func()


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 norm_layer='default',
                 act_func='default',
                 bias=False
                 ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.Identity
        elif norm_layer == 'default':
            norm_layer = nn.BatchNorm2d

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=groups,
                padding=(kernel_size - 1) // 2,
                bias=bias
            ),
            norm_layer(out_channels),
            getActFunc(act_func)
        )

    def forward(self, x):
        return self.block(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=4, act_func='default'):
        super().__init__()
        reduced_dim = int(in_channels / reduction)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channels, out_channels=reduced_dim, kernel_size=1),
            getActFunc(act_func),
            nn.Conv2d(in_channels=reduced_dim, out_channels=in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class InvertedResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 expand_ratio=1,
                 reduction=4,
                 survival_prob=.8,
                 act_func='default',
                 norm_layer='default'
                 ):
        super().__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        if self.use_residual:
            self.stochastic_dropout = StochasticDropout(survival_prob)

        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        if self.expand:
            self.expand_conv = ConvBlock(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=1,
                act_func=act_func,
                norm_layer=norm_layer
            )

        self.convSq = nn.Sequential(
            ConvBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                groups=hidden_dim,
                act_func=act_func,
                norm_layer=norm_layer
            ),
            SqueezeExcitation(in_channels=hidden_dim, reduction=reduction, act_func=act_func),
            ConvBlock(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1,
                      act_func=None, norm_layer=norm_layer),
        )

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        x = self.convSq(x)

        if self.use_residual:
            x = inputs + self.stochastic_dropout(x, self.training)
        return x


class FusedMBConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expand_ratio,
                 survival_prob,
                 kernel_size=3,
                 stride=1,
                 reduction=4,
                 norm_layer='default',
                 act_func='default',

                 ):
        super().__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        if self.use_residual:
            self.stochastic_dropout = StochasticDropout(survival_prob)

        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim

        if self.expand:
            self.expand_conv = ConvBlock(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                norm_layer=norm_layer,
                act_func=act_func
            )
        self.convSq = ConvBlock(
            in_channels=hidden_dim,
            out_channels=out_channels,
            kernel_size=1 if self.expand else kernel_size,
            stride=1 if self.expand else stride,
            norm_layer=norm_layer,
            act_func=None if self.expand else act_func
        )

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        x = self.convSq(x)

        if self.use_residual:
            x = inputs + self.stochastic_dropout(x, self.training)
        return x

MBConv = InvertedResidualBlock