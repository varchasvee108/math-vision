import os
import json
import random
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import argparse
from data.processor import VOCAB, stoi, itos, PAD_ID, EOS_ID
from core.config import Config


def generate_single_image(size: tuple, max_digits=4):

    width, height = size
    img = Image.new("L", size=size, color=255)
    draw = ImageDraw.Draw(img)

    a = random.randint(0, 999)
    b = random.randint(0, 999)
    result = a + b

    var_text = f"a = {a}\n b = {b}"
    query_text = "Find: a + b"

    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except:
        font = ImageFont.load_default()

    rand_x = random.randint(10, 40)
    rand_y = random.randint(10, 40)

    draw.text((rand_x, rand_y), var_text, fill=0, font=font)
    center_y = height // 2 + random.randint(-10, 10)
    draw.text((rand_x, center_y), query_text, fill=0, font=font)

    answer_str = str(result)
    tokens = [stoi[s] for s in answer_str]
    tokens.append(EOS_ID)

    while len(tokens) < max_digits + 1:
        tokens.append(PAD_ID)

    return img, tokens, result


def generate_raw_data():
    config = Config.load()
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=config.data.seed)
    args = parser.parse_args()

    random.seed(args.seed)

    raw_dir = Path(config.data.raw_dir)
    split_dir = Path(config.data.split_dir)

    raw_dir.mkdir(exist_ok=True, parents=True)
    split_dir.mkdir(exist_ok=True, parents=True)

    dataset = []

    image_size = tuple(config.data.image_size)

    for i in range(args.samples):
        img, tokens, result = generate_single_image(image_size)

        file_name = f"math_{i:06d}.png"
        file_path = raw_dir / file_name
        img.save(file_path)

        dataset.append(
            {"image_path": str(file_path), "target_ids": tokens, "label_value": result}
        )

        if (i + 1) % 500 == 0:
            print(f"Generated {i + 1} samples")

    random.shuffle(dataset)
    split_point = int(0.8 * len(dataset))

    train_dataset = dataset[:split_point]
    val_dataset = dataset[split_point:]

    with open(split_dir / "train.json", "w") as f:
        json.dump(train_dataset, f, indent=4)

    with open(split_dir / "val.json", "w") as f:
        json.dump(val_dataset, f, indent=4)

    print(f"Generated {args.samples} samples")
    print("Train/Val split: ", len(train_dataset), len(val_dataset))
    print("Saved to", raw_dir, split_dir)


if __name__ == "__main__":
    generate_raw_data()
