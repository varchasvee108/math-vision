import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from data.processor import VisionMathProcessor


class VisionMathDataset(Dataset):
    def __init__(self, config, processor: VisionMathProcessor, split="train"):
        self.processor = processor

        path = f"{config.data.split_dir}/{split}.json"
        with open(path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        img = Image.open(data["image_path"]).convert("L")
        pixel_values = self.processor.preprocess_image(img)
        labels = torch.tensor(data["target_ids"], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}
