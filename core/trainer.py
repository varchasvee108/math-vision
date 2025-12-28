import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.processor import VisionMathProcessor
from models.model import MathVision
from core.config import Config
from data.dataloader import create_dataloader
from transformers import get_scheduler
from data.processor import PAD_ID


class Trainer:
    def __init__(
        self,
        config: Config,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        processor: VisionMathProcessor,
        optimizer,
        pad_id: int,
        device: torch.device,
        grad_clip: int = 0,
        scheduler=None,
        scaler=None,
        loss_fn=None,
    ):
        self.config = config
        self.model = model.to(device=device)
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

    @classmethod
    def setup_trainer(cls, config: Config):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        processor = VisionMathProcessor(image_size=config.data.image_size)
        model = MathVision(config=config)
        train_dataloader, val_dataloader = create_dataloader(
            config=config, processor=processor
        )

        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2 and not name.endswith("bias"):
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optimizer = AdamW(
            [
                {"params": decay_params, "weight_decay": config.training.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=config.training.lr,
            betas=tuple(config.training.betas),
        )
        scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=config.training.warmup_steps,
            num_training_steps=config.training.max_steps,
        )

        loss_fn = F.cross_entropy
        scaler = torch.amp.GradScaler()
        return cls(
            config=config,
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            processor=processor,
            optimizer=optimizer,
            pad_id=PAD_ID,
            device=device,
            grad_clip=config.training.grad_clip,
            scheduler=scheduler,
            scaler=scaler,
            loss_fn=loss_fn,
        )

    def train_step(self, batch):
        self.model.train()

        images = batch["pixel_values"].to(self.device)
        labels = batch["labels"].to(self.device)

        decoder_inputs = labels[:, :-1]
        decoder_targets = labels[:, 1:]

        with torch.autocast(
            device_type=self.device.type, enabled=(self.device.type == "cuda")
        ):
            logits = self.model(images, decoder_inputs)
            loss = self.loss_fn(
                logits.reshape(-1, logits.size(-1)), decoder_targets.reshape(-1)
            )

        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if self.grad_clip is not None and self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.grad_clip is not None and self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        return loss.item()

    def evaluate(self):
        self.model.eval()
        self.model.eval()
        total_loss = 0.0

        for batch in self.val_dataloader:
            pixels = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)

            decoder_inputs = labels[:, :-1]
            target_labels = labels{:, 1:}
            logits = self.model(pixels, decoder_inputs)
            loss = self.loss_fn(logits.reshape(-1, logits.size(-1)))


    def train(self):
        print(
            f"Starting training on {self.device.type} device for {self.config.training.max_steps} steps..."
        )

        train_iter = iter(self.train_dataloader)
        val_iter = iter(self.val_dataloader)
        pbar = tqdm(total=self.config.training.max_steps)

        for step in tqdm(range(self.config.training.max_steps)):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)
            loss = self.train_step(batch)
            pbar.update(1)
            pbar.set_description(f"Loss: {loss:.4f}")

            if step % self.config.training.val_every_steps == 0:
                val_loss = self.evaluate()
                print(
                    f"\n {step} steps | Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f}"
                )

            if step % self.config.training.vis_every_steps == 0:
                self.visualize(step)
        pbar.close()
        print("Training complete")
