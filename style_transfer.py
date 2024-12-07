import argparse
import torch
import torchvision
import math
import matplotlib.pyplot as plt

from src.utils import load_image, img_show
from src.optimize import run_optim
from src.constants import device, weight_style, weight_content


def main(style_path: str, content_path: str, output_path: str, initial_image: str):
    torch.cuda.empty_cache()
    # The images have been loaded for you.
    style_image = load_image(style_path)
    content_image = load_image(content_path)

    # Center crop the style_image to match the dimensions of the content image.
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

    # Display the original input image: (style image)
    plt.figure()
    img_show(style_image, title="Style Img")

    # Display the content image:
    plt.figure()
    img_show(content_image, title="Content Img")

    cnn = torchvision.models.vgg19(pretrained=True).features.to(device).eval()

    if initial_image == "content":
        input_image = content_image.clone()
    elif initial_image == "noise":
        input_image = torch.rand_like(content_image, device=device)
    else:
        raise ValueError("Invalid initial image type")

    output = run_optim(
        cnn,
        content_image,
        style_image,
        input_image,
        number_steps=300,
        weight_style=weight_style,
        weight_content=weight_content,
        use_content=True,
        use_style=True,
    )

    fig = plt.figure()
    img_show(output, title="Output Image")
    save_image(fig, output_path)


def save_image(fig, path: str):
    fig.savefig(path)
    print(f"Saved {path}")


parser = argparse.ArgumentParser()

parser.add_argument("--content", type=str, required=True)
parser.add_argument("--style", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument(
    "--initial_image", type=str, choices=["content", "noise"], default="content"
)


if __name__ == "__main__":
    args = parser.parse_args()

    main(args.style, args.content, args.output, args.initial_image)
