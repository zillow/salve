from torch import nn
from torchvision import models


def get_resnet_feature_dim(num_layers: int) -> int:
    """ """
    if num_layers in [18, 34]:
        feature_dim = 512 * 1  # expansion factor 1
    elif num_layers in [50, 101, 152]:
        feature_dim = 512 * 4  # expansion factor 4, so get 2048
    else:
        raise RuntimeError("Num layers not allowed")

    return feature_dim


def get_vanilla_resnet_model(num_layers: int, pretrained: bool) -> nn.Module:
    assert num_layers in [18, 34, 50, 101, 152]
    if num_layers == 18:
        resnet = models.resnet18(pretrained=pretrained)
    elif num_layers == 34:
        resnet = models.resnet34(pretrained=pretrained)
    elif num_layers == 50:
        resnet = models.resnet50(pretrained=pretrained)
    else:
        raise RuntimeError("num layers not supported")

    return resnet
