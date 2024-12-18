import torch

from src.setup_model import get_model_and_losses
from src.constants import content_default_layers, style_default_layers


def run_optim(
    cnn: torch.nn.Module,
    content_image: torch.Tensor,
    style_images: list[torch.Tensor],
    input_image: torch.Tensor,
    steps: int,
    optimizer: torch.optim.LBFGS,
    style_weight: torch.types.Number,
    content_weight: torch.types.Number,
):
    model, style_loss_errors, content_loss_errors = get_model_and_losses(
        cnn, style_images, content_image, content_default_layers, style_default_layers
    )
    input_image.requires_grad_()
    run_step = {"count": 0}

    style_losses = []
    content_losses = []

    def closure():
        optimizer.zero_grad()
        model(input_image)

        total_style_loss = torch.stack(
            tuple(map(lambda err: err.loss, style_loss_errors))
        ).sum(0)
        total_style_loss = total_style_loss.mul(style_weight)

        total_content_loss = torch.stack(
            tuple(map(lambda err: err.loss, content_loss_errors))
        ).sum(0)
        total_content_loss = total_content_loss.mul(content_weight)

        loss = total_style_loss + total_content_loss
        loss.backward()

        run_step["count"] += 1

        style_losses.append(total_style_loss.item())
        content_losses.append(total_content_loss.item())

        print("\033[K" + f"Step: {run_step['count']}, Loss: {loss}", end="\r")
        return loss

    while run_step["count"] <= steps:
        optimizer.step(closure)

        with torch.no_grad():
            input_image.clamp_(0, 1)

    print("\n")

    return input_image, (style_losses, content_losses)
