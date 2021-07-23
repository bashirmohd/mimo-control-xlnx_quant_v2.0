import torch.nn as nn
from stable_baselines3.common.torch_layers import create_mlp

nn_configs = {
    "3x3": {
        "input_dim": 25,
        "output_dim": 8,
        "net_arch": [60, 30, 15]
    },
    "9x9_center9": {
        "input_dim": 121,
        "output_dim": 9,
        "net_arch": [80, 50, 25]
    },
    "9x9": {
        "input_dim": 289,
        "output_dim": 81,
        "net_arch": [400, 200, 100]
    }
}


def create_recognizer(
    double_frame: bool,
    nn_config: str
) -> nn.Sequential:
    if nn_config in nn_configs:
        config = nn_configs[nn_config]
    else:
        raise NotImplementedError()
    if double_frame:
        config["input_dim"] *= 2
    layers = create_mlp(**config)
    return nn.Sequential(*layers)
