import argparse
import openai

import src.prompts as prompts
import src.generation as generation


parser = argparse.ArgumentParser()
parser.add_argument("--lyrics", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--image_size", type=str, default="1024x1024")
parser.add_argument("--artist", type=str, default=None)
parser.add_argument("--max_retries", type=int, default=3)
parser.add_argument("--log_description", action="store_true")
parser.add_argument("--log_url", action="store_true")
parser.add_argument("--magical_atmosphere", action="store_true")
parser.add_argument("--include_intricate_details", action="store_true")


def generate_image_from_lyrics(
    song_lyrics: str,
    log_description=False,
    log_url=False,
    image_size="1024x1024",
    max_retries=3,
    magical_atmosphere=False,
    include_intricate_details=False,
    artist: str | None = None,
):
    prompt = prompts.get_scene_description_prompt(
        song_lyrics, magical_atmosphere, include_intricate_details, artist
    )

    for i in range(max_retries):
        try:
            scene_description = generation.generate_text(prompt)
            if log_description:
                print(f"Scene description: {scene_description}", end="\n\n")
            image = generation.generate_image(
                f"{scene_description}\n\nDo not include words in the generated image.",
                image_size,
                log_url,
            )
            break
        except openai.BadRequestError as error:
            if i == max_retries - 1:
                print("Image generation rejected. Max number of retries exceeded!")
                raise error

            print("Image generation rejected for content policy violation. Retrying...")
            prompt = prompts.get_scene_fixing_prompt(scene_description)

    return image


if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.lyrics, "r", encoding="utf-8") as file:
        song_lyrics = file.read()

    output = generate_image_from_lyrics(
        song_lyrics,
        args.log_description,
        args.log_url,
        args.image_size,
        args.max_retries,
        args.magical_atmosphere,
        args.include_intricate_details,
        args.artist,
    )

    with open(args.output, "wb") as file:
        file.write(output)
    print(f"Image saved successfully at {args.output}")
