from torch.utils.data import DataLoader
from data.processor import VisionMathProcessor
from core.config import Config


def create_dataloader(config: Config, processor: VisionMathProcessor):
    train_dataset = VisionMathProcessor(config.data.image_size[0])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    val_dataset = VisionMathProcessor(config.data.image_size[0])
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    return train_dataloader, val_dataloader
