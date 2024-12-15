import argparse
import json

import src.prompts as prompts
import src.generation as generation


parser = argparse.ArgumentParser()
parser.add_argument("--lyrics", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--log_description", action="store_true")
parser.add_argument("--log_url", action="store_true")


def generate_video_from_lyrics(lyrics: str, log_description=False, log_url=False):
    prompt = prompts.get_video_description_prompt(lyrics)

    frames_description = generation.generate_text(prompt)
    if log_description:
        print(f"Scene description: {frames_description}", end="\n\n")
    scene_frames: dict[str, str] = json.loads(frames_description)

    return generation.generate_video(scene_frames, log_url)


if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.lyrics, "r", encoding="utf-8") as file:
        lyrics = file.read()

    video_bytes = generate_video_from_lyrics(lyrics, args.log_description, args.log_url)

    with open(args.output, "wb") as file:
        file.write(video_bytes)
    print(f"Video saved successfully at {args.output}")
