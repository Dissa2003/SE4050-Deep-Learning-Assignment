"""Dataset loading and preprocessing utilities."""

from pathlib import Path
from typing import Callable, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


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

        self.samples = []
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
