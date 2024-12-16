import argparse
import torch
import torchvision
import matplotlib.pyplot as plt

from src.utils import save_image, ImageDataset, get_image_optimizer, resize_style_image
from src.optimize import run_optim
from src.constants import device


def style_transfer(
    style_paths: list[str],
    content_path: str,
    output_path: str,
    initial_image: str,
    image_size: list[int] | None,
    steps: int,
    style_weight: float,
    content_weight: float,
    plot_loss: bool,
):
    if torch.cuda.is_available():
        print("Models moved to GPU.")
        torch.cuda.empty_cache()

    dataset = ImageDataset(image_size)

    content_image = dataset[content_path]
    style_images = list(
        map(lambda path: resize_style_image(dataset[path], content_image), style_paths)
    )

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

    optimizer = get_image_optimizer(input_image)

    output_image, (style_losses, content_losses) = run_optim(
        cnn,
        content_image,
        style_images,
        input_image,
        steps,
        optimizer,
        style_weight,
        content_weight,
    )

    if plot_loss:
        figure, ax1 = plt.subplots()
        ax1.set_xlabel("Steps")

        (line1,) = ax1.plot(style_losses, "b-", label="Style loss")
        ax1.set_ylabel("Style loss (blue)")

        ax2 = ax1.twinx()
        (line2,) = ax2.plot(content_losses, "g-", label="Content loss")
        ax2.set_ylabel("Content loss (green)")

        lines = [line1, line2]
        figure.legend(lines, [line.get_label() for line in lines])  # type: ignore
        plt.tight_layout()
        plt.show()

    save_image(output_image, output_path)


parser = argparse.ArgumentParser()

parser.add_argument("--content", type=str, required=True)
parser.add_argument("--style", type=str, nargs="+", required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument(
    "--initial_image", type=str, choices=["content", "noise"], default="content"
)
parser.add_argument("--image_size", type=int, nargs="+", default=None)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--style_weight", type=float, default=200000.0)
parser.add_argument("--content_weight", type=float, default=1.0)
parser.add_argument("--plot_loss", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()

    style_transfer(
        args.style,
        args.content,
        args.output,
        args.initial_image,
        args.image_size,
        args.steps,
        args.style_weight,
        args.content_weight,
        args.plot_loss,
    )
