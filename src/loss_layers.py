import torch


class Content_Error_Loss(torch.nn.Module):
    def __init__(self, target: torch.Tensor):
        super(Content_Error_Loss, self).__init__()

        self.target = target.detach()
        self.criterion = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor):
        self.loss = self.criterion(x, self.target)
        return x


def gram_matrix(activations: torch.Tensor):
    a, b, c, d = activations.size()

    features = activations.view(a * b, c * d)
    gram = features @ features.T
    normalized_gram = gram / (b * c * d)

    return normalized_gram


class Style_Error_Loss(torch.nn.Module):
    def __init__(self, target: torch.Tensor):
        super(Style_Error_Loss, self).__init__()

        self.target_gram = gram_matrix(target).detach()
        self.criterion = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor):
        input_gram = gram_matrix(x)
        self.loss = self.criterion(input_gram, self.target_gram)

        return x
