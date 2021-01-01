import torch
import torch.nn as nn
from torchvision.models import resnet34


class ResNet34(nn.Module):
    """
    ResNet34 for wave image classification
    classes: binary (PORT, STBD)

    :param num_classes: 1 for regression
    :param pretrained: resnet pretrained
    :param fc_bias: bias in last fully connected layer
    """

    def __init__(
        self, num_classes: int, pretrained: bool = False, fc_bias: bool = False
    ):
        super(ResNet34, self).__init__()
        self.resnet = resnet34(pretrained=pretrained)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        modules = list(self.resnet.children())[:-1]

        self.extractor = nn.Sequential(*modules)
        self.classifier = nn.Linear(
            self.resnet.fc.in_features, num_classes, bias=fc_bias
        )
        self.resnet = None

    def forward(self, batch):
        feature = self.extractor(batch)
        feature = feature.view(feature.size(0), -1)
        logits = self.classifier(feature)
        return logits
