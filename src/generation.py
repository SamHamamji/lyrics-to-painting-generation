import requests
import openai


def generate_text(prompt: str):
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are an expert in analyzing song lyrics for artistic interpretation.",
            },
            {"role": "user", "content": prompt},
        ],
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
