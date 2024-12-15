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


if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.lyrics, "r", encoding="utf-8") as file:
        song_lyrics = file.read()

    prompt = prompts.get_scene_description_prompt(
        song_lyrics,
        args.magical_atmosphere,
        args.include_intricate_details,
        args.artist,
    )

    for i in range(args.max_retries):
        try:
            scene_description = generation.generate_text(prompt)
            if args.log_description:
                print(f"Scene description: {scene_description}", end="\n\n")
            image = generation.generate_image(
                f"{scene_description}\n\nDo not include words in the generated image.",
                args.image_size,
                args.log_url,
            )
            break
        except openai.BadRequestError as e:
            if i == args.max_retries - 1:
                print("Image generation rejected. Max number of retries exceeded!")
                exit(1)

            print("Image generation rejected for content policy violation. Retrying...")
            prompt = prompts.get_scene_fixing_prompt(scene_description)

    with open(args.output, "wb") as file:
        file.write(image)
    print(f"Image saved successfully at {args.output}")
