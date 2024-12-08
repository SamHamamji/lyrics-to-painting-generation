import argparse
import torch
import torchvision
import math

from src.utils import save_image, ImageDataset
from src.optimize import run_optim
from src.constants import device, style_weight, content_weight


def resize_style_image(style_image: torch.Tensor, content_image: torch.Tensor):
    if style_image.size(dim=1) != 3:
        style_image = style_image.repeat(1, 3, 1, 1)
    if content_image.size(dim=3) > style_image.size(dim=3):
        pad_size = math.ceil(
            (content_image.size(dim=3) - style_image.size(dim=3)) / 2.0
        )
        padding = torchvision.transforms.Pad(
            (pad_size, pad_size), padding_mode="reflect"
        )
        style_image = padding(style_image)
    if content_image.size(dim=2) > style_image.size(dim=2):
        pad_size = math.ceil(
            (content_image.size(dim=2) - style_image.size(dim=2)) / 2.0
        )
        padding = torchvision.transforms.Pad(
            (0, 0, pad_size, pad_size), padding_mode="reflect"
        )
        style_image = padding(style_image)

    centerCrop = torchvision.transforms.CenterCrop(
        (content_image.size(dim=2), content_image.size(dim=3))
    )
    style_image = centerCrop(style_image)

    assert (
        style_image.size() == content_image.size()
    ), "We need to import the style and content images at the same size."

    return style_image


def main(
    style_path: str,
    content_path: str,
    output_path: str,
    initial_image: str,
    image_size: list[int],
    steps: int,
):
    if torch.cuda.is_available():
        print("Models moved to GPU.")
        torch.cuda.empty_cache()

    dataset = ImageDataset(image_size)

    style_image = dataset[style_path]
    content_image = dataset[content_path]
    style_image = resize_style_image(style_image, content_image)

    cnn = (
        torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
        .features.to(device)
        .eval()
    )

    if initial_image == "content":
        input_image = content_image.clone()
    elif initial_image == "noise":
        input_image = torch.rand_like(content_image, device=device)
    else:
        raise ValueError("Invalid initial image type")

    output_image = run_optim(
        cnn,
        content_image,
        style_image,
        input_image,
        steps=steps,
        style_weight=style_weight,
        content_weight=content_weight,
        use_content=True,
        use_style=True,
    )

    save_image(output_image, output_path)


parser = argparse.ArgumentParser()

parser.add_argument("--content", type=str, required=True)
parser.add_argument("--style", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument(
    "--initial_image", type=str, choices=["content", "noise"], default="content"
)
parser.add_argument("--image_size", type=int, nargs="+", default=256)
parser.add_argument("--steps", type=int, default=300)


if __name__ == "__main__":
    args = parser.parse_args()

    main(
        args.style,
        args.content,
        args.output,
        args.initial_image,
        args.image_size,
        args.steps,
    )
