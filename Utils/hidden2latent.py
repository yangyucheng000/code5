import torch
import torch.nn as nn
import math


class Hidden2Latent(nn.Module):
    def __init__(self, num_linear, input_features, output_features, bias=True):
        super(Hidden2Latent, self).__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(num_linear,
                                                input_features,
                                                output_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear, output_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        k = 1.0 / self.input_features
        bound = math.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_x: torch.Tensor):
        # [d, n, q] = [d, n, m1] @ [d, m1, q]
        out = torch.matmul(input_x, self.weight)

        if self.bias is not None:
            # [d, n, q] += [d, 1, q]
            out += self.bias.unsqueeze(dim=1)
        return out
