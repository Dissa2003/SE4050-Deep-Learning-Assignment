"""Dataset loading, splitting, and preprocessing utilities for Brain Tumor MRI classification."""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
RANDOM_SEED = 42


class BrainTumorDataset(Dataset):
    """Custom PyTorch Dataset for loading brain tumor MRI images.

    Expects a directory structure of: root_dir/<class_name>/<image>.jpg
    """

    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted(
            [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        )
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples: List[Tuple[Path, int]] = []
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            for img_path in cls_dir.glob("*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class TransformSubset(Dataset):
    """Wraps a `Subset` of `BrainTumorDataset` to apply a split-specific transform.

    This lets train/val/test share the same underlying file index while using
    different preprocessing pipelines (e.g. augmentation only on the train split).
    """

    def __init__(self, subset: Subset, transform: Callable):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        original_index = self.subset.indices[index]
        img_path, label = self.subset.dataset.samples[original_index]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label


def get_transforms(train: bool = True) -> transforms.Compose:
    """Return the preprocessing/augmentation pipeline for train or eval splits.

    Train: Resize -> RandomRotation -> RandomHorizontalFlip -> ColorJitter -> Normalize.
    Val/Test: Resize -> Normalize only (no augmentation, kept unseen/clean).
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def split_dataset(
    data_dir: str,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = RANDOM_SEED,
) -> Tuple[Dataset, Dataset, Dataset, List[str]]:
    """Load the raw dataset and split into train (70%) / val (15%) / test (15%).

    A fixed random seed ensures the split is reproducible across runs, and the
    test subset only ever receives non-augmenting transforms so it stays a
    clean, completely unseen evaluation set.
    """
    base_dataset = BrainTumorDataset(data_dir, transform=None)
    total = len(base_dataset)

    test_size = int(total * test_split)
    val_size = int(total * val_split)
    train_size = total - val_size - test_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        base_dataset, [train_size, val_size, test_size], generator=generator
    )

    train_ds = TransformSubset(train_subset, get_transforms(train=True))
    val_ds = TransformSubset(val_subset, get_transforms(train=False))
    test_ds = TransformSubset(test_subset, get_transforms(train=False))

    return train_ds, val_ds, test_ds, base_dataset.classes


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = RANDOM_SEED,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Build train/val/test DataLoaders for the Brain Tumor MRI dataset."""
    train_ds, val_ds, test_ds, classes = split_dataset(data_dir, val_split, test_split, seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, classes
