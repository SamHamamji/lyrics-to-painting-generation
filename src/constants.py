import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

content_default_layers = ["conv_4"]
style_default_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]
