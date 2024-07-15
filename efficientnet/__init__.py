import torch.nn as nn
from dataclasses import dataclass
import dataclasses
from functools import partial
import math

from .model import EfficientNet
from cls import ModelLoader

v1_base_topology = [
    # expand_ratio, in_channels, out_channels, repeats, stride, kernel_size, conv_type, reduction
    [1, 32, 16, 1, 1, 3, 1, 4],
    [6, 16, 24, 2, 2, 3, 1, 4],
    [6, 24, 40, 2, 2, 5, 1, 4],
    [6, 40, 80, 3, 2, 3, 1, 4],
    [6, 80, 112, 3, 1, 5, 1, 4],
    [6, 112, 192, 4, 2, 5, 1, 4],
    [6, 192, 320, 1, 1, 3, 1, 4]
]

v1_amplification_factors = {
    # tuple :phi_value, resolution_range, drop_rate
    "b0": (0, (224, 224), .2),
    "b1": (.5, (240, 240), .2),
    "b2": (1, (260, 260), .3),
    "b3": (2, (300, 300), .3),
    "b4": (3, (380, 380), .4),
    "b5": (4, (456, 456), .4),
    "b6": (5, (528, 528), .5),
    "b7": (6, (600, 600), .5),
}

v2_topylogies = {
    "S": [
        # expand_ratio, in_channels, out_channels, repeats, stride, kernel_size, conv_type, reduction
        [1, 24, 24, 2, 1, 3, 0, 0],
        [4, 24, 48, 4, 2, 3, 0, 0],
        [4, 48, 64, 4, 2, 3, 0, 0],
        [4, 64, 128, 6, 2, 3, 1, 4],
        [6, 128, 160, 9, 1, 3, 1, 4],
        [6, 160, 256, 15, 2, 3, 1, 4]
    ],
    "M": [
        # expand_ratio, in_channels, out_channels, repeats, stride, kernel_size, conv_type, reduction
        [1, 24, 24, 3, 1, 3, 0, 0],
        [4, 24, 48, 5, 2, 3, 0, 0],
        [4, 48, 80, 5, 2, 3, 0, 0],
        [4, 80, 160, 7, 2, 3, 1, 4],
        [6, 160, 176, 14, 1, 3, 1, 4],
        [6, 176, 304, 18, 2, 3, 1, 4],
        [6, 304, 512, 5, 1, 3, 1, 4]
    ],
    "L": [
        # expand_ratio, in_channels, out_channels, repeats, stride, kernel_size, conv_type, reduction
        [1, 32, 32, 4, 1, 3, 0, 0],
        [4, 32, 64, 7, 2, 3, 0, 0],
        [4, 64, 96, 7, 2, 3, 0, 0],
        [4, 96, 192, 10, 2, 3, 1, 4],
        [6, 192, 224, 19, 1, 3, 1, 4],
        [6, 224, 384, 25, 2, 3, 1, 4],
        [6, 384, 640, 7, 1, 3, 1, 4]
    ]
}

v2_amplification_factors = {
    # resolution_range,dropout
    "S": ((300, 384), 0.2),
    "M": ((384, 480), 0.3),
    "L": ((384, 480), 0.4)
}


@dataclass
class EfficientNetConfig:
    squeeze_function: object
    norm_layer: object
    version: str
    size: str
    net_topology: list
    resolution_range: tuple
    affine_layers: int = 1280  # fully connection neurons
    act_func: str = 'SILU'
    survival_prob: float = 0.8
    reduction: int = 4
    dropout: float = 0.2
    n_embd: int = 512  # dimension of classification vector
    img_layers: int = 3

    def serialize(self):
        return dataclasses.asdict(self)


class EfficientNetLoader(ModelLoader):
    def __init__(self, path=None, **options):
        if path is not None:
            options = {
                **self.load_options(path),
                **options,
            }

        assert options.get('version', None) is not None \
               and options.get('size', None) is not None, \
            "initialize EfficientNet must specify version and size parameters," \
            "for example version = 'V1' and size = 'b0'"

        v = int(options['version'][-1])
        if v == 1:
            opt_dict = self.calculate_factors(options['size'].lower())
            reduction = 4 if 'reduction' not in options else options['reduction']
            net_topology, dropout = self.generate_frame(model_factors=opt_dict['model_factors'], reduction=reduction)
            opt_dict.pop('model_factors')
            options.update(opt_dict)

        elif v == 2:
            net_topology = v2_topylogies[options['size'].upper()]
            resolution_range, dropout = v2_amplification_factors[options['size'].upper()]
            options['resolution_range'] = resolution_range

        options['net_topology'] = net_topology
        options['dropout'] = dropout

        if 'squeeze_function' not in options:
            options['squeeze_function'] = nn.AdaptiveAvgPool2d(1)
        else:
            if isinstance(options['squeeze_function'], str):
                self.squeeze_function = getattr(nn, options['squeeze_function'], nn.AdaptiveAvgPool2d)(1)
            else:
                self.squeeze_function = (nn.AdaptiveAvgPool2d(1)
                                         if options['squeeze_function'] is None
                                         else options['squeeze_function'])

        if 'norm_layer' not in options:
            options['norm_layer'] = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        self.config = EfficientNetConfig(**options)

    def load_model(self):
        return EfficientNet(self.config)

    def out_opts(self):
        print(self.serial_opts(self.config))

    def generate_frame(self, model_factors, reduction=4):
        width_factor, depth_factor, drop_rate = model_factors
        net_topology = []

        for block in v1_base_topology:
            # modify the params of the base topology
            copy_block = [*block]
            copy_block[1] = reduction * math.ceil(int(copy_block[1] * width_factor) / reduction)  # in_channels
            copy_block[2] = reduction * math.ceil(int(copy_block[2] * width_factor) / reduction)  # out_channels
            copy_block[3] = math.ceil(copy_block[3] * depth_factor)  # repeats
            net_topology.append(copy_block)

        return net_topology, drop_rate

    def calculate_factors(self, size, alpha=1.2, beta=1.1):
        phi, resolution, drop_rate = v1_amplification_factors[size]
        depth_factor = alpha ** phi
        width_factor = beta ** phi

        return {
            'model_factors': [width_factor, depth_factor, drop_rate],
            'affine_layers': math.ceil(1280 * width_factor),
            'resolution_range': resolution
        }
