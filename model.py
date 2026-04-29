import torch
import timm

def get_model(model_name="resnet50", num_classes=10, pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model