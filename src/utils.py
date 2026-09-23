"""Training, evaluation, and EDA/baseline analysis helper functions."""

import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from skimage.feature import hog
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across python/numpy/torch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# EDA helpers
# --------------------------------------------------------------------------

def get_class_distribution(dataset, classes: List[str]) -> Dict[str, int]:
    """Count samples per class label for a `BrainTumorDataset`-like object."""
    labels = [label for _, label in dataset.samples]
    counts = Counter(labels)
    return {classes[idx]: count for idx, count in sorted(counts.items())}


def get_image_resolutions(dataset, sample_size: int = 200) -> List[Tuple[int, int]]:
    """Sample raw images (before resizing) and return their (width, height)."""
    n = min(sample_size, len(dataset.samples))
    indices = np.random.choice(len(dataset.samples), size=n, replace=False)
    resolutions = []
    for idx in indices:
        img_path, _ = dataset.samples[idx]
        with Image.open(img_path) as img:
            resolutions.append(img.size)
    return resolutions


def plot_class_distribution(class_counts: Dict[str, int]) -> None:
    """Plot a bar chart and pie chart of class distribution side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), ax=axes[0])
    axes[0].set_title("Class Distribution (Count)")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Number of Images")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].pie(class_counts.values(), labels=class_counts.keys(), autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Class Distribution (Proportion)")

    plt.tight_layout()
    plt.show()


def plot_sample_grid(dataset, classes: List[str], n_rows: int = 4, n_cols: int = 4) -> None:
    """Display an n_rows x n_cols grid of random sample images with class labels."""
    n_samples = n_rows * n_cols
    indices = np.random.choice(len(dataset.samples), size=n_samples, replace=False)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))
    for ax, idx in zip(axes.flatten(), indices):
        img_path, label = dataset.samples[idx]
        image = Image.open(img_path).convert("RGB")
        ax.imshow(image)
        ax.set_title(classes[label], fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def detect_class_imbalance(class_counts: Dict[str, int], threshold: float = 1.5) -> None:
    """Print a simple imbalance ratio report (max class count / min class count)."""
    counts = list(class_counts.values())
    ratio = max(counts) / min(counts)
    print(f"Class imbalance ratio (max/min): {ratio:.2f}")
    if ratio >= threshold:
        print("Potential class imbalance detected — consider class weighting or resampling.")
    else:
        print("Class distribution is reasonably balanced.")


# --------------------------------------------------------------------------
# Classical ML baseline helpers
# --------------------------------------------------------------------------

def get_samples(dataset) -> List[Tuple[Path, int]]:
    """Return (image_path, label) pairs for a `BrainTumorDataset` or `TransformSubset`."""
    if hasattr(dataset, "samples"):
        return dataset.samples
    if hasattr(dataset, "subset"):  # TransformSubset
        base_samples = dataset.subset.dataset.samples
        return [base_samples[i] for i in dataset.subset.indices]
    raise TypeError(f"Unsupported dataset type: {type(dataset)}")


def extract_flatten_features(dataset, image_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Extract flattened, normalized grayscale pixel features for an ML baseline."""
    features, labels = [], []
    for img_path, label in get_samples(dataset):
        image = Image.open(img_path).convert("L").resize((image_size, image_size))
        features.append(np.asarray(image, dtype=np.float32).flatten() / 255.0)
        labels.append(label)
    return np.array(features), np.array(labels)


def extract_hog_features(dataset, image_size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """Extract HOG (Histogram of Oriented Gradients) features for an ML baseline."""
    features, labels = [], []
    for img_path, label in get_samples(dataset):
        image = Image.open(img_path).convert("L").resize((image_size, image_size))
        hog_features = hog(
            np.asarray(image),
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
        )
        features.append(hog_features)
        labels.append(label)
    return np.array(features), np.array(labels)


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, classes: List[str], title: str = "Model"
) -> Dict[str, float]:
    """Compute accuracy/precision/recall/F1 and plot a confusion matrix."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    print(f"--- {title} Evaluation ---")
    for name, value in metrics.items():
        print(f"{name.capitalize()}: {value:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, xticks_rotation=30, cmap="Blues", colorbar=False)
    ax.set_title(f"{title} - Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return metrics


# --------------------------------------------------------------------------
# PyTorch training / evaluation
# --------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run a single training epoch and return the average loss."""
    model.train()
    running_loss = 0.0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model and return (average loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()

    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 10,
) -> Dict[str, List[float]]:
    """Train for multiple epochs, printing and tracking loss/accuracy history."""
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} "
            f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    return history


@torch.no_grad()
def get_predictions(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference over a dataloader and return (y_true, y_pred) numpy arrays."""
    model.eval()
    all_labels, all_preds = [], []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def save_checkpoint(model: nn.Module, path: str) -> None:
    """Save model weights to the given path, creating parent dirs if needed."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

