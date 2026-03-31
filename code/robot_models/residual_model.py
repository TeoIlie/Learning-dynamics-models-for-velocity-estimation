import torch
import numpy as np


class ResidualNeuralModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, res_mlp_size: int, res_layer_count: int, res_activation):
        super(ResidualNeuralModel, self).__init__()
        self.model = model

        def create_nn(input_size, output_size, mlp_size, layer_count,
                      activation):
            layers = [torch.nn.Linear(input_size, mlp_size), activation()]
            for _ in range(layer_count - 2):
                layers.append(torch.nn.Linear(mlp_size, mlp_size))
                layers.append(activation())
            layers.append(torch.nn.Linear(mlp_size, output_size))
            return torch.nn.Sequential(*layers)

        self.nn = create_nn(7, 4, res_mlp_size,
                            res_layer_count,
                            res_activation)

    def forward(self, t, x):
        x_dot = torch.concat([self.nn(x),
                              torch.zeros(x.shape[:-1] + (3,), device=x.device, dtype=x.dtype)], dim=-1)
        return x_dot + self.model(t, x)
