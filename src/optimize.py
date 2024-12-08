import torch

from src.setup import get_model_and_losses


def run_optim(
    cnn: torch.nn.Module,
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    input_image: torch.Tensor,
    steps: int,
    style_weight: torch.types.Number,
    content_weight: torch.types.Number,
    use_content=True,
    use_style=True,
):
    model, style_loss_errors, content_loss_errors = get_model_and_losses(
        cnn, style_image, content_image
    )
    input_image.requires_grad_()
    optimizer = torch.optim.LBFGS([input_image])
    run_step = {"count": 0}

    def closure():
        optimizer.zero_grad()
        model(input_image)

        total_style_loss = (
            torch.stack(tuple(map(lambda err: err.loss, style_loss_errors))).sum(0)
            if use_style
            else torch.zeros(())
        )

        total_content_loss = (
            torch.stack(tuple(map(lambda err: err.loss, content_loss_errors))).sum(0)
            if use_content
            else torch.zeros(())
        )

        loss = total_style_loss * style_weight + total_content_loss * content_weight
        loss.backward()

        run_step["count"] += 1

        print("\033[K" + f"Step: {run_step['count']}, Loss: {loss}", end="\r")
        return loss

    while run_step["count"] <= steps:
        optimizer.step(closure)

        with torch.no_grad():
            input_image.clamp_(0, 1)

    print("\n")

    return input_image
