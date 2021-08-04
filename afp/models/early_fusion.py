"""
Early-fusion ResNet architecture.
"""

from typing import Optional, Tuple

import torch
from torch import nn, Tensor

from afp.models.resnet_factory import get_vanilla_resnet_model, get_resnet_feature_dim


class EarlyFusionCEResnet(nn.Module):
    """ """

    def __init__(self, num_layers: int, pretrained: bool, num_classes: int, args) -> None:
        super(EarlyFusionCEResnet, self).__init__()
        assert num_classes > 1

        self.modalities = args.modalities

        self.resnet = get_vanilla_resnet_model(num_layers, pretrained)
        self.inplanes = 64

        num_inchannels = 3 * 4  # four RGB images
        # resnet with more channels in first layer
        self.conv1 = nn.Conv2d(num_inchannels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        feature_dim = get_resnet_feature_dim(num_layers)
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        x3: Optional[Tensor],
        x4: Optional[Tensor],
        x5: Optional[Tensor],
        x6: Optional[Tensor],
    ) -> torch.Tensor:
        """ """
        if self.modalities == ["layout"]:
            x = torch.cat([x1, x2], dim=1)
        elif set(self.modalities) == set(["ceiling_rgb_texture", "floor_rgb_texture"]):
            x = torch.cat([x1, x2, x3, x4], dim=1)
        elif set(self.modalities) == set(["ceiling_rgb_texture", "floor_rgb_texture", "layout"]):
            x = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)
        else:
            raise RuntimeError(f"Unsupported modalities. {str(self.modalities)}")

        x = self.conv1(x)

        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        logits = self.fc(x)

        return logits
