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


video_generation_prompt = """
Given the below song lyrics, generate a series of 5 temporal frame prompts to be fed to a text-to-video model in order to generate an animated PAINTING.

```
{lyrics}
```

Put emphasis on the KEY PARTS that DISTINGUISH the song with a nice description.
Don't include irrelevant details that might distract the model's attention, focus on visual hints.
Don't use complex grammar, if the song has a transition trend or progression showcase it.
Take the key most important features for each independent frame.
Don't include inappropriate description at all!

Frame 1 should be the scene intro, frame 5 is the outro, etc...
Your output should in json format with keys being frame_1, frame_2, ... the items being the actual prompt for the corresponding frame.
"""


def get_scene_description_prompt(
    lyrics: str, magical_atmosphere: bool, include_intricate_details: bool, artist: str
):
    extra_instructions = []
    if magical_atmosphere:
        extra_instructions.append("Include a magical and enchanting atmosphere.")
    if include_intricate_details:
        extra_instructions.append(
            "Ensure the scene contains intricate details and textures."
        )
    if artist:
        extra_instructions.append(
            f"Style the painting in the artistic style of {artist}."
        )

    extra_instructions_text = " ".join(extra_instructions)
    return scene_description_prompt.format(
        lyrics=lyrics, extra_instructions=extra_instructions_text
    ).strip()


def get_scene_fixing_prompt(scene: str):
    return scene_fixing_prompt.format(scene=scene)


def get_video_generation_prompt(lyrics: str):
    return video_generation_prompt.format(lyrics=lyrics)
