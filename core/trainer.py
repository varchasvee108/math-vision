import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        config: Config,
        model: torch.nn.Module,
        train_dataloader: Dataloader,
        val_dataloader: Dataloader,
        processor: VisionMathProcessor,
        optimizer,
        pad_id: int,
        device: torch.device = "cuda",
        grad_clip: int = 0,
        scheduler=None,
        scaler=None,
        loss_fn=None,
    ):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.processor = processor
        self.device = device
        self.scheduler = scheduler
        self.scaler = scaler
        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.loss_fn = loss_fn or F.cross_entropy
        self.pad_id = pad_id

        self.model.to(self.device)
