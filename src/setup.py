import sys
import time
import copy
import math

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from src.loss_layers import Content_Error_Loss, Style_Error_Loss
from src.utils import Normalization

from src.constants import (
    cnn_norm_mean,
    cnn_norm_std,
    device,
    content_default_layers,
    style_default_layers,
)


def get_model_and_losses(
    cnn_model,
    style_image,
    content_image,
    content_layers=content_default_layers,
    style_layers=style_default_layers,
):
    cnn_model = copy.deepcopy(cnn_model)

    content_loss_errors = []
    style_loss_errors = []

    #############################
    ### Your code starts here ###
    #############################
    normalization = Normalization(mean=cnn_norm_mean, std=cnn_norm_std).to(device)
    model = torch.nn.Sequential(normalization)

    layer_num = 0

    for layer in cnn_model.children():
        if len(content_loss_errors) == len(content_layers) and len(
            style_loss_errors
        ) == len(style_layers):
            break

        if isinstance(layer, torch.nn.Conv2d):
            layer_num += 1
            model.append(layer)

            name = f"conv_{layer_num}"
            if name in content_layers:
                content_loss = Content_Error_Loss(model(content_image).detach())
                model.append(content_loss)
                content_loss_errors.append(content_loss)

            if name in style_layers:
                style_loss = Style_Error_Loss(model(style_image).detach())
                model.append(style_loss)
                style_loss_errors.append(style_loss)

        elif isinstance(layer, torch.nn.ReLU):
            model.append(torch.nn.ReLU(inplace=False))

        else:
            model.append(layer)

    #############################
    ### Your code ends here   ###
    #############################

    return model, style_loss_errors, content_loss_errors
