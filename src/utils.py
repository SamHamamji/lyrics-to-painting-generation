import torch
import torchvision
import PIL.Image
import matplotlib.pyplot as plt

from src.constants import device, imgsize

loader = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(imgsize),
        torchvision.transforms.ToTensor(),
    ]
)

unloader = torchvision.transforms.ToPILImage()

if torch.cuda.is_available():
    print("Models moved to GPU.")


def load_image(path: str):
    image = PIL.Image.open(path)

    image = torch.Tensor(loader(image)).unsqueeze(0)
    return image.to(device, torch.float)


def img_show(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


class Normalization(torch.nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super(Normalization, self).__init__()

        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img: torch.Tensor):
        return (img - self.mean) / self.std


def get_img_optimizer(input_image):

    optimizer = torch.optim.LBFGS([input_image])
    return optimizer
