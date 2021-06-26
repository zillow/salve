

from torchvision import models


def get_vanilla_resnet_model(num_layers: int, pretrained: bool) -> nn.Module
	assert num_layers in [18,34,50,101,152]
	if num_layers == 18:
		resnet = models.resnet18(pretrained=pretrained)
	elif num_layers == 34:
		resnet = models.resnet34(pretrained=pretrained)
	elif num_layers == 50:
		resnet = models.resnet50(pretrained=pretrained)
	else:
		raise RuntimeError("num layers not supported")

	return resnet