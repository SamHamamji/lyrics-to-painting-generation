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


def get_scene_description_prompt(
    lyrics: str,
    magical_atmosphere: bool,
    include_intricate_details: bool,
    style_by_artist: str,
):
    extra_instructions = []
    if magical_atmosphere:
        extra_instructions.append("Include a magical and enchanting atmosphere.")
    if include_intricate_details:
        extra_instructions.append(
            "Ensure the scene contains intricate details and textures."
        )
    if style_by_artist:
        extra_instructions.append(
            f"Style the painting in the artistic style of {style_by_artist}."
        )

    extra_instructions_text = " ".join(extra_instructions)
    return generating_scene_prompt.format(
        lyrics=lyrics, extra_instructions=extra_instructions_text
    ).strip()


def get_scene_fixing_prompt(scene: str):
    return fixing_scene_prompt.format(scene=scene)
