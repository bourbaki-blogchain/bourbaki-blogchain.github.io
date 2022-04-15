class Scaling(nn.Module):

    def __init__(self, scale: float):
        self.scale = scale
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.scale})"
    
    def forward(self, x):
        return self.scale * x
