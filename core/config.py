import pathlib
import sys
from typing import List
from dataclasses import dataclass
import tomllib


@dataclass(frozen=true)
class ProjectConfig:
    name: str
    experiment_name: str


@dataclass(frozen=true)
class DataConfig:
    raw_dir: str
    split_dir: str
    batch_size: int
    num_workers: int
    image_size: list[int]
    seed: int


@dataclass(frozen=true)
class ModelConfig:
    architecture: str
    num_layers: int
    dropout: float
    dim_ratio: int


@dataclass(frozen=true)
class TrainingConfig:
    lr: float
    max_steps: int
    warmup_steps: int
    weight_decay: float
    betas: list[float]
    grad_clip: float
    val_every_steps: int
    vis_every_steps: int
    scheduler: str


@dataclass(frozen=true)
class LoggingConfig:
    use_wandb: bool
    project_name: str
    assets_dir: str


@dataclass(frozen=true)
class Config:
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig

    @classmethod
    def load(cls, path: str = "configs/base.toml"):
        path_obj = pathlib.Path(path)
        if not path_obj.exists and not path_obj.is_file():
            raise FileNotFoundError(f"Config file {path} does not exist at path {path}")

        with open(path_ob, "rb") as f:
            data = tomllib.load(f)

        return cls(
            project=ProjectConfig(**data["project"]),
            data=DataConfig(**data["data"]),
            model=ModelConfig(**data["model"]),
            training=TrainingConfig(**data["training"]),
            logging=LoggingConfig(**data["logging"]),
        )
