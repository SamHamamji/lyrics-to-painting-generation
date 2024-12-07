import torch

from src.setup import get_model_and_losses


def run_optim(
    cnn,
    content_image,
    style_image,
    input_image,
    number_steps: int,
    weight_style: torch.types.Number,
    weight_content: torch.types.Number,
    use_content=True,
    use_style=True,
):
    #############################
    ### Your code starts here ###
    #############################

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
            sum(map(lambda style_loss: style_loss.loss, style_loss_errors))
            * weight_style
            if use_style
            else torch.zeros(())
        )

        total_content_loss = (
            sum(map(lambda content_loss: content_loss.loss, content_loss_errors))
            * weight_content
            if use_content
            else torch.zeros(())
        )

        total_loss = total_style_loss + total_content_loss
        total_loss.backward()

        run_step["count"] += 1

        return total_loss

    while run_step["count"] <= number_steps:
        optimizer.step(closure)

        with torch.no_grad():
            input_image.clamp_(0, 1)

    #############################
    ### Your code ends here   ###
    #############################

    return input_image
