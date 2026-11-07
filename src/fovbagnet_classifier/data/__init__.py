"""Dataset and dataloader construction utilities.

This module provides factory functions for constructing ImageNet datasets
and PyTorch dataloaders with standardized training and validation
transform pipelines. It centralizes all data preprocessing logic,
including augmentation, resizing, and tensor conversion, to ensure
consistent experimental setup across runs.
"""

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision.transforms import v2


def get_dataset(cfg, mode="train"):
    """Create an ImageNet dataset with mode-specific transforms.

    Builds a torchvision ImageNet dataset with either training or
    validation preprocessing. Training mode applies random augmentations,
    while validation mode uses deterministic resizing and cropping.

    Args:
        cfg: Configuration object containing dataset path, image size,
            and augmentation settings.
        mode: Dataset split to use. Must be either "train" or "val".

    Returns:
        A torchvision ImageNet dataset instance.
    """
    image_resolution = cfg.data.image_resolution
    validation_crop_ratio = cfg.data.augmentation.validation_crop_ratio
    use_color_jitter = cfg.data.augmentation.use_color_jitter

    validation_resize_resolution = round(image_resolution / validation_crop_ratio)

    transforms = (
        [
            v2.RandomResizedCrop(image_resolution),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4) if use_color_jitter else None,
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
        if mode == "train"
        else [
            v2.Resize(validation_resize_resolution),
            v2.CenterCrop(image_resolution),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    transform = v2.Compose([f for f in transforms if f is not None])

    image_dataset = ImageNet(
        root=cfg.data.path_to_image_root,
        split=mode,
        transform=transform,
    )
    return image_dataset


def get_dataloader(cfg, mode="train"):
    """Create a DataLoader for training or validation.

    Wraps the dataset with PyTorch DataLoader using configuration-defined
    batch size and worker settings. Training loaders enable shuffling and
    drop the last incomplete batch, while validation loaders are
    deterministic.

    Args:
        cfg: Configuration object containing dataloader parameters.
        mode: Dataset split to use. Must be either "train" or "val".

    Returns:
        A PyTorch DataLoader instance.
    """
    mode_is_train = mode == "train"
    return DataLoader(
        dataset=get_dataset(cfg, mode),
        shuffle=mode_is_train,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=2,
        persistent_workers=cfg.data.persistent_workers,
        drop_last=mode_is_train,
    )
