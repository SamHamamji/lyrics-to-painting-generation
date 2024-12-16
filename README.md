# Painting generation from lyrics

## Setup
- Add your Novita and OpenAI API keys to the `.env` file
- Add your Novita and OpenAI API keys to the `.env` file
- Download and unzip both datasets

This step requires that your `.kaggle.json` file is set up accordingly.
Note that the size of the two datasets sums up to 2.76 GB, using custom data works fine too.
```bash
./setup_datasets.sh
```

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

## Credits
Credits for the datasets are given to:
- [ikarus777/best-artworks-of-all-time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time)
- [neisse/scrapped-lyrics-from-6-genres](https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres)

## License
This work is licensed under the [MIT license](./LICENSE).
