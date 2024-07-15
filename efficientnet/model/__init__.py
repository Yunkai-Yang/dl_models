import torch
import torch.nn as nn

from .block import MBConv, FusedMBConv, ConvBlock

ConvType = [FusedMBConv, MBConv]


class EfficientNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        connecting_nn = self._create_connecting_nn(config=config)
        self.initial = connecting_nn['initial']
        self.bottleneck = connecting_nn['bottleneck']
        self.affine_layer = connecting_nn['affine_layer']
        self.backbone = self._create_backbone(config=config)

        self._init_weight()

    def _create_backbone(self, config):
        act_func = config.act_func
        if act_func.upper() == 'SILU':
            act_func = 'default'

        norm_layer = config.norm_layer

        total_layers = sum([i[3] for i in config.net_topology])
        deactivation_prob = 1 - config.survival_prob
        layer_depth = 0
        layer_blocks = []

        for expand_ratio, in_channels, out_channels, repeats, stride, kernel_size, conv_type, reduction in config.net_topology:
            for i in range(repeats):
                layer_blocks.append(
                    ConvType[conv_type](
                        in_channels=in_channels if i == 0 else out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        expand_ratio=expand_ratio,
                        stride=stride if i == 0 else 1,
                        reduction=reduction,
                        survival_prob=1 - (deactivation_prob * layer_depth / total_layers),
                        act_func=act_func,
                        norm_layer=norm_layer
                    )
                )
                layer_depth += 1

        return nn.Sequential(*layer_blocks)

    def _create_connecting_nn(self, config):
        dropout_rate = config.dropout
        act_func = config.act_func
        squeeze_function = config.squeeze_function
        if act_func.upper() == 'SILU':
            act_func = 'default'

        norm_layer = config.norm_layer
        init_channels, last_channels = config.img_layers, config.affine_layers
        bottleneck = nn.Sequential(
            ConvBlock(
                in_channels=config.net_topology[-1][2],
                out_channels=last_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                act_func=act_func,
            ),
            nn.Dropout(dropout_rate),
            squeeze_function,
            nn.Flatten()
        )
        return {
            'initial': ConvBlock(
                in_channels=init_channels,
                out_channels=config.net_topology[0][1],
                stride=2,
                act_func=act_func,
                norm_layer=norm_layer),
            'bottleneck': bottleneck,
            'affine_layer': nn.Linear(last_channels, config.n_embd)
        }

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.initial(x)
        x = self.backbone(x)
        x = self.bottleneck(x)
        return self.affine_layer(x)
