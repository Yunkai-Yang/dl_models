import dataclasses
from dataclasses import dataclass

from .model import Transformer
from cls import ModelLoader


@dataclass
class TransformerConfig:
    block_size: int = 1024
    vocab_size: int = 1024
    enc_layer: int = 3
    dec_layer: int = 3
    n_head: int = 8
    n_embd: int = 512
    pe: bool = True  # whether to use a learnable position embedding
    dropout: float = 0.1
    bias: bool = True
    act_func: str = 'relu'


class TransformerLoader(ModelLoader):
    def __init__(self, path=None, **options):
        if path is not None:
            options = {
                **self.load_options(path),
                **options,
            }

        self.config = TransformerConfig(**options)

    def load_model(self):
        return Transformer(self.config)

    def out_opts(self):
        print(self.serial_opts(self.config))
