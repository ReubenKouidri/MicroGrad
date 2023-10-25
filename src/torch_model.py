import torch.nn as nn
from collections import OrderedDict


class TorchMLP(nn.Sequential):
    def __init__(self, input_size, output_sizes):
        layers = [nn.Linear(input_size if i == 0 else output_sizes[i - 1], output_sizes[i]) for i in
                  range(len(output_sizes))]

        topology = OrderedDict()
        for i, layer in enumerate(layers):
            topology[f"layer_{i}"] = layer
        topology["activation"] = nn.Tanh()
        super().__init__(topology)
