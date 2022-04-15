from torch import nn
from typing import Callable


class PreResidualBottleneck(nn.Module):

    expansion = 4

    def __init__(self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[[int], nn.Module] = None,
        no_preact: bool = False,  # additional argument
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        ### pre-activations ###
        preact_layers = [] if no_preact else [
            norm_layer(inplanes),
            nn.ReLU(),
        ]
        if downsample is None:
            self.preact = nn.Identity()
            residual_preact = preact_layers
        else:
            self.preact = nn.Sequential(*preact_layers)
            residual_preact = []
        ### pre-activations ###

        kernel_size = 3
        width = groups * (planes * base_width // 64)
        self.downsample = nn.Identity() if downsample is None else downsample
        self.residual_branch = nn.Sequential(
            *residual_preact,  # include residual pre-activations
            nn.Conv2d(inplanes, width, 1, bias=False),
            norm_layer(width), nn.ReLU(),
            nn.Conv2d(width, width, kernel_size, stride, padding=dilation,
                      dilation=dilation, groups=groups, bias=False),
            norm_layer(width), nn.ReLU(),
            nn.Conv2d(width, planes * self.expansion, 1, bias=False),
            # norm_layer(planes * self.expansion),
        )

    def forward(self, x):
        x = self.preact(x)  # compute global pre-activations
        skip = self.downsample(x)
        residual = self.residual_branch(x)
        # return torch.relu(residual + skip)
        return residual + skip
