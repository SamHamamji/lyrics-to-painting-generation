import torch
import torchvision
import PIL.Image

from src.constants import device


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_size: list[int]):
        self.loader = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(image_size),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __getitem__(self, path: str):
        image = PIL.Image.open(path)

        image_tensor = torch.Tensor(self.loader(image)).unsqueeze(0)
        return image_tensor.to(device, torch.float)

    def __len__(self):
        return 0


def save_image(tensor: torch.Tensor, path: str):
    np_tensor = (tensor.detach() * 255).byte().squeeze(0).permute(1, 2, 0).numpy()
    image = PIL.Image.fromarray(np_tensor)
    image.save(path)
    print(f"Saved {path}")


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
