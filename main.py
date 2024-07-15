from transformer import TransformerLoader
from efficientnet import EfficientNetLoader

if __name__ == '__main__':
    import torch
    import torch.nn as nn

    device = "cuda"
    # x = torch.randn(1, 5, 512).to(device)
    # target = torch.randint(0, 10, (1, 5)).to(device)
    # padding_mask = torch.ones((1, 5, 1), dtype=torch.bool).to(device)
    # model = TransformerLoader(n_head = 4).load_model().to(device)
    # print(model(x, target, padding_mask).shape)
    # TransformerLoader(n_head = 4).out_opts()
    x = torch.randn(3, 3, 224,224).to(device)
    loader = EfficientNetLoader(version='v1', size='b5')
    m =loader.load_model().to(device)
    print(m)
    x = m(x)
    print(x.shape)
