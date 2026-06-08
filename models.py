import torch.nn as nn
import timm
from torchvision import models

import config


def build_model(arch, num_classes, *, dropout=None, pretrained=None):
    use_pretrained = config.PRETRAINED if pretrained is None else pretrained
    drop = config.DROPOUT if dropout is None else dropout
    w = models.ResNet50_Weights.IMAGENET1K_V2 if use_pretrained else None
    if arch == "cnn":
        m = models.resnet50(weights=w)
        m.fc = nn.Sequential(
            nn.Dropout(p=drop),
            nn.Linear(m.fc.in_features, num_classes),
        )
        return m
    m = timm.create_model("vit_base_patch16_224", pretrained=use_pretrained, num_classes=num_classes)
    return m


def replace_head(model, num_classes, arch, *, dropout=None):
    drop = config.DROPOUT if dropout is None else dropout
    if arch == "cnn":
        in_f = model.fc[1].in_features
        model.fc = nn.Sequential(nn.Dropout(p=drop), nn.Linear(in_f, num_classes))
        return model
    in_f = model.head.in_features
    model.head = nn.Linear(in_f, num_classes)
    return model
