from torch.utils.data import DataLoader
from datasets.dataset import VisionMathDataset
from core.config import Config
from data.processor import VisionMathProcessor


def create_dataloader(config: Config, processor: VisionMathProcessor):
    train_dataset = VisionMathDataset(config, processor, split="train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    val_dataset = VisionMathDataset(config, processor, split="val")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    return train_dataloader, val_dataloader
