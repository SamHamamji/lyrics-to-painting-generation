# Painting generation from lyrics

## Setup
Add your Novita and OpenAI API keys to the `.env` file.

## Usage
Image generation:
```bash
python lyrics_to_image.py --lyrics ./sample_lyrics.txt --output ./out.png
```

Style transfer:
```bash
python style_transfer.py --content ./content.jpg --style ./style1.jpg ./style2.jpg ... --output ./out.jpg
```

Video generation:
```bash
python lyrics_to_image.py --lyrics ./sample_lyrics.txt --output ./out.mp4
```

For all arguments, use `--help`.

## Example script for using dataset
This example selects 3 random Picasso paintings as styles.
```bash
python lyrics_to_image.py --lyrics ./sample_lyrics.txt --output ./out.png --artist "Pablo Picasso"
python style_transfer.py --style $(ls ./images/images/Pablo_Picasso/* | shuf | head -n 3) --content content/painting_from_lyrics.png --output output/temp.jpg --steps 200
```

## License
This work is licensed under the [MIT license](./LICENSE).
