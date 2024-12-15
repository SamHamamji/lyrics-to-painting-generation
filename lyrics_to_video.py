import argparse
import json

import src.prompts as prompts
import src.generation as generation


parser = argparse.ArgumentParser()
parser.add_argument("--lyrics", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--log_description", action="store_true")
parser.add_argument("--log_url", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.lyrics, "r", encoding="utf-8") as file:
        song_lyrics = file.read()

    prompt = prompts.get_video_generation_prompt(song_lyrics)

    frames_description = generation.generate_text(prompt)
    if args.log_description:
        print(f"Scene description: {frames_description}", end="\n\n")
    scene_frames: dict[str, str] = json.loads(frames_description)

    video_bytes = generation.generate_video(scene_frames, args.log_url)

    with open(args.output, "wb") as file:
        file.write(video_bytes)
    print(f"Video saved successfully at {args.output}")
