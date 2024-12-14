import argparse
import os
import openai
import requests
import dotenv


parser = argparse.ArgumentParser()
parser.add_argument("--lyrics_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--max_retries", type=int, default=3)
parser.add_argument("--image_size", type=str, default="1024x1024")
parser.add_argument("--log_description", action="store_true")
parser.add_argument("--log_image_url", action="store_true")
parser.add_argument("--magical_atmosphere", action="store_true")
parser.add_argument("--include_intricate_details", action="store_true")
parser.add_argument("--style_by_artist", type=str, default=None)


generating_scene_prompt = """
Provide a brief, vivid, scene description to be fed for a painting generation tool. The scene should capture the mood, themes, and imagery of the following lyrics:

'{lyrics}'

Don't include inappropriate content at all!
"""

fixing_scene_prompt = """
The following prompt has been rejected because it contains some text that violates an image generation content policy.

'{scene}'

Fix this scene description by ONLY tweaking the problematic part(s) and returning the rest WORD FOR WORD.
"""

def get_scene_description_prompt(lyrics: str, magical_atmosphere: bool, include_intricate_details: bool, style_by_artist: str):
    extra_instructions = []
    if magical_atmosphere:
        extra_instructions.append("Include a magical and enchanting atmosphere.")
    if include_intricate_details:
        extra_instructions.append("Ensure the scene contains intricate details and textures.")
    if style_by_artist:
        extra_instructions.append(f"Style the painting in the artistic style of {style_by_artist}.")
    
    extra_instructions_text = " ".join(extra_instructions)
    return generating_scene_prompt.format(lyrics=lyrics, extra_instructions=extra_instructions_text).strip()


def fix_scene_description_prompt(scene: str):
    return fixing_scene_prompt.format(scene=scene)


def generate_text(prompt: str):
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are an expert in analyzing song lyrics for artistic interpretation."},
                  {"role": "user", "content": prompt}],
    )

    description = response.choices[0].message.content

    if description is None:
        raise ValueError(f"Scene description is None in response: {response}")

    return description.strip()


def generate_image(prompt: str, image_size, log_image_url: bool):
    response = openai.images.generate(
        prompt=prompt, model="dall-e-3", n=1, size=image_size
    )

    image_url = response.data[0].url

    if image_url is None:
        raise ValueError(f"Image url is None in response: {response}")

    if log_image_url:
        print(f"Image url: {image_url}")

    response = requests.get(image_url, timeout=120)

    if response.content is None:
        raise ValueError(f"Image content is None in response: {response}")

    return response.content


if __name__ == "__main__":
    args = parser.parse_args()

    dotenv.load_dotenv()
    openai.api_key = os.environ["OPENAI_API_KEY"]

    with open(args.lyrics_path, "r", encoding="utf-8") as file:
        song_lyrics = file.read()
        
    prompt = get_scene_description_prompt(
            song_lyrics, 
            args.magical_atmosphere, 
            args.include_intricate_details, 
            args.style_by_artist
        )

    for i in range(args.max_retries):
        try:
            scene_description = generate_text(prompt)
            if args.log_description:
                print(f"Scene description: {scene_description}", end="\n\n")
            image = generate_image(
                f"{scene_description}\n\nDo not include words in the generated image.",
                args.image_size,
                args.log_image_url,
            )
            break
        except openai.BadRequestError as e:
            if i == args.max_retries - 1:
                print("Image generation rejected. Max number of retries exceeded!")
                exit(1)

            print(
                "Image generation rejected for content policy violation. Retrying...",
                end="\n\n",
            )
            prompt = fix_scene_description_prompt(scene_description)

    with open(args.output_path, "wb") as file:
        file.write(image)
    print(f"Image saved successfully at {args.output_path}")
