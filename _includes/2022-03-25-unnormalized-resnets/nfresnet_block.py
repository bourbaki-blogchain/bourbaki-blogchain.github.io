from torch import nn
from typing import Callable


class NFResidualBottleneck(nn.Module):

    expansion = 4

    def __init__(self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        alpha: float = 1.,
        beta: float = 1.,
        no_preact: bool = False,
    ):
        super().__init__()
        self.beta = beta

        ### pre-activations ###
        preact_layers = [] if no_preact else [
            Scaling(alpha),
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
            *residual_preact,
            nn.Conv2d(inplanes, width, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(width, width, kernel_size, stride, padding=dilation,
                      dilation=dilation, groups=groups, bias=True),
            nn.ReLU(),
            nn.Conv2d(width, planes * self.expansion, 1, bias=True),
        )

    def forward(self, x):
        x = self.preact(x)
        skip = self.downsample(x)
        residual = self.residual_branch(x)
        return self.beta * residual + skip
    
    @torch.no_grad()
    def signal_prop(self, x, dim=(0, -1, -2)):
        # forward code
        x = self.preact(x)
        skip = self.downsample(x)
        residual = self.residual_branch(x)
        out = self.beta * residual + skip

        # compute necessary statistics
        out_mu2 = torch.mean(out.mean(dim) ** 2).item()
        out_var = torch.mean(out.var(dim)).item()
        res_var = torch.mean(residual.var(dim)).item()
        return out, (out_mu2, out_var, res_var)
