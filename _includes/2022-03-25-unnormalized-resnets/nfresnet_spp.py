class NFResidualNetwork(nn.Module):

    @staticmethod
    def initialisation(m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def __init__(self, layers: tuple, num_classes: int = 1000, beta: float = 1.):
        super().__init__()
        block = NFResidualBottleneck
        self._inplanes = 64
        self._expected_var = 1.
        self.beta = beta

        self.intro = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.subnet1 = self._make_subnet(block, 64, layers[0], no_preact=True)
        self.subnet2 = self._make_subnet(block, 128, layers[1], stride=2)
        self.subnet3 = self._make_subnet(block, 256, layers[2], stride=2)
        self.subnet4 = self._make_subnet(block, 512, layers[3], stride=2)

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512 * block.expansion, num_classes),
        )

        self.apply(self.initialisation)
        # self.apply(CentredWeightNormalization(dim=(1, 2, 3)))
    
    def _make_subnet(self, block, planes: int, num_layers: int, 
                     stride: int = 1, no_preact: bool = False):
        downsample = None
        if stride != 1 or self._inplanes != planes * block.expansion:
            downsample = nn.Conv2d(self._inplanes, planes * block.expansion, 1, stride)
        
        layers = []
        # compute expected variance analytically
        alpha = 1. / self._expected_var ** .5
        self._expected_var = 1. + self.beta ** 2
        layers.append(block(
            self._inplanes, planes, stride, downsample,
            alpha=alpha, beta=self.beta, no_preact=no_preact
        ))
        self._inplanes = planes * block.expansion
        for _ in range(1, num_layers):
            # track expected variance analytically
            alpha = 1. / self._expected_var ** .5
            self._expected_var += self.beta ** 2
            layers.append(block(
                self._inplanes, planes, alpha=alpha, beta=self.beta
            ))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.intro(x)
        x = self.subnet1(x)
        x = self.subnet2(x)
        x = self.subnet3(x)
        x = self.subnet4(x)
        return self.classifier(x)

    @torch.no_grad()
    def signal_prop(self, x, dim=(0, -1, -2)):
        x = self.intro(x)

        statistics = [(
            torch.mean(x.mean(dim) ** 2).item(),
            torch.mean(x.var(dim)).item(),
            float('nan'),
        )]
        for subnet in (self.subnet1, self.subnet2, self.subnet3, self.subnet4):
            for layer in subnet:
                x, stats = layer.signal_prop(x, dim)
                statistics.append(stats)
        
        # convert list of tuples to tuple of lists
        sp = tuple(map(list, zip(*statistics)))
        return self.classifier(x), sp
