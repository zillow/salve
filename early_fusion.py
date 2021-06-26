

from typing import Tuple

import torch
from torch import nn, Tensor

from resnet_factory import get_vanilla_resnet_model

class EarlyFusionCEResnet(nn.Module):
    """ """
    def __init__(self, num_layers: int, pretrained: bool, num_classes: int, args) -> None:
        super(EarlyFusionCEResnet, self).__init__()
        assert num_classes > 1

        resnet = get_vanilla_resnet_model(num_layers, pretrained)
        self.inplanes = 64

        # resnet with more channels in first layer
        self.conv1 = nn.Conv2d(
            num_inchannels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )

    def forward(self, x1: Tensor, x2: Tensor, x3: Tensor, x4: Tensor) -> torch.Tensor:
        """ """
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        logits = self.fc(x)

        return logits
