"""
Main training script for the GRU-ODE-Bayes model on the PhysioNet dataset.
This script is for the Eager-Mode optimized model and includes a complete
checkpointing system to save and resume training.
"""

# src.train_val.train_model

import torch
from typing import Any
from tqdm import tqdm
from pathlib import Path
from torch import nn, optim
from torch.utils.data import DataLoader


from src.model.ode_model_corpus import GRUODEBayes
from src.data_preproc.config import PhysioNetConfig
from src.data_preproc.dataset import PhysioNetDataset
from src.data_preproc.collators import collate_function

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = PhysioNetConfig()
FINAL_DATA_DIR = CONFIG.final_data_dir
CHECKPOINT_DIR = CONFIG.check_point_dir
CHECKPOINT_PATH_LATEST = CHECKPOINT_DIR / "latest.pth"
CHECKPOINT_PATH_BEST = CHECKPOINT_DIR / "best.pth"

MODEL_PARAMS = {
    "input_size": 7,
    "hidden_size": 64,
    "p_hidden_size": 32,
    "prep_hidden_size": 32,
    "cov_size": 3,
    "cov_hidden_size": 16,
    "class_hidden_size": 64,
    "class_output_size": 3,
    "bias": True,
    "dropout_rate": 0.0,
    "mixing": 1e-1,
    "full_gru_ode": True,
    "solver": "euler",
    "rtol": 1e-3,
    "atol": 1e-4,
    "fixed_step_size": False,
}


def train_model(
    device: torch.device = DEVICE,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    num_epochs: int = 10,
    class_weight: float = 1e-1,
    grad_clip_val: float = 1.0,
) -> None:
    """
    Training function that loads models parameters from saved staff if any,
    uses saved processed data, creates check points for each epoch, saving the new
    paramets if they have show any improvement.
    """
    total_loss: torch.Tensor
    print("Initializing Training...")
    # --- Data Loading ---
    train_data_list, val_data_list = load_preproc_data(device)

    # --- Initializng Datasets and Loaders ---
    train_loader, val_loader = init_dataloader(
        train_data_list, val_data_list, batch_size, device
    )

    # --- Model, Optimizer and Criterion setup ---
    print("Intializing Model, Optimizer and Criterion.")
    model = GRUODEBayes(**MODEL_PARAMS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_criterion = nn.CrossEntropyLoss()

    # --- Loads from checkpoint if any ---
    start_epoch, best_val_accuracy = load_checkpoint(model, optimizer, device)

    # --- Training loop ---
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n--- Epoch {epoch:_}/{num_epochs:_} ---")
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for batch in pbar:
            batch_on_device = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            optimizer.zero_grad(set_to_none=True)
            h, total_loss, class_pred, class_loss_total = model(
                class_criterion=class_criterion,
                class_weight=class_weight,
                **batch_on_device,  # Batch is properly formated, since we use a `custom_collate_fn`
            )
            # Pytorch autograd magic
            total_loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)  # To be safe...
            optimizer.step()
            total_train_loss += total_loss.item()
            pbar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                class_loss=f"{class_loss_total.item():.4f}",
            )

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        # --- Validation ---
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_samples = 0

        # --- NEW: Add tqdm progress bar for the validation loop ---
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch} Validation")

        with torch.no_grad():
            for batch in val_pbar:  # Iterate over the progress bar
                batch_on_device = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                h, total_loss, class_pred, class_loss_total = model(
                    class_criterion=class_criterion,
                    class_weight=class_weight,
                    **batch_on_device,
                )
                total_val_loss += total_loss.item()
                _, predicted_labels = torch.max(class_pred, 1)
                labels_on_device = batch_on_device["labels"]
                correct_predictions += (
                    (predicted_labels == labels_on_device).sum().item()
                )
                total_samples += labels_on_device.size(0)

                # Update the validation progress bar postfix
                val_pbar.set_postfix(val_loss=f"{total_loss.item():.4f}")

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = (correct_predictions / total_samples) * 100
        print(f"Average Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        # --- Checkpoint ---
        best_val_accuracy = save_checkpoint(
            epoch, model, optimizer, avg_val_loss, val_accuracy, best_val_accuracy
        )
    print("\n--- Training Complete ---")


def load_preproc_data(
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    print(
        (
            f"Checkpoints will be loaded from and saved to: {CHECKPOINT_DIR}\n"
            f"Using device: {device}\n"
            f"Loading pre-process data from: {FINAL_DATA_DIR}\n"
        )
    )
    train_data_list = torch.load(
        FINAL_DATA_DIR / "train_chunks.pt", map_location="cpu", weights_only=False
    )
    val_data_list = torch.load(
        FINAL_DATA_DIR / "val_chunks.pt", map_location="cpu", weights_only=False
    )
    print("Pre-Processed Data Properly Loaded.")
    return train_data_list, val_data_list


def init_dataloader(
    train_data_list: list[dict[str, Any]],
    val_data_list: list[dict[str, Any]],
    batch_size: int,
    device: torch.device,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    print("Creating Data Loaders...")
    train_dataset = PhysioNetDataset(train_data_list)
    val_dataset = PhysioNetDataset(val_data_list)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_function,
        num_workers=4,
        pin_memory=True if device == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_function,
        num_workers=4,
    )
    print(
        (
            "Data Loader created.\n",
            f"Using Batch Size of: {batch_size} Units.",
        )
    )

    return train_loader, val_loader


def load_checkpoint(
    model: nn.Module, optimizer: optim.Optimizer, device: torch.device
) -> tuple[int, float]:
    start_epoch = 0
    best_val_accuracy = 0.0
    if CHECKPOINT_PATH_LATEST.exists():
        print(f"Resuming training from checkpoint: {CHECKPOINT_PATH_LATEST}")
        checkpoint = torch.load(
            CHECKPOINT_PATH_LATEST, map_location=device
        )  # I dont known if i need to include `weights_only=False`
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1  # Do i need to send to CPU ?
        best_val_accuracy = checkpoint.get("best_val_accuracy", 0.0)
        print(
            f"Resumed from epoch {start_epoch -1}\n"
            f"Best validation accuracy so far: {best_val_accuracy:.3f}\n"
        )
    else:
        print("No checkpoint found. Starting training from scratch.")

    return start_epoch, best_val_accuracy


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    avg_val_loss: float,
    val_accuracy: float,
    best_val_accuracy: float,
) -> float:
    # --- CHECKPOINTING ---
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_val_loss,
            "accuracy": val_accuracy,
            "best_val_accuracy": best_val_accuracy,
        },
        CHECKPOINT_PATH_LATEST,
    )
    print(f"Saved latest model checkpoint to {CHECKPOINT_PATH_LATEST}")
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_val_loss,
                "accuracy": val_accuracy,
            },
            CHECKPOINT_PATH_BEST,
        )
        print(
            f"*** New best model saved with validation accuracy: {val_accuracy:.2f}% ***"
        )
    return best_val_accuracy


# Run this file if you want, but beware to initialize the other modules expected in `.data/` folder.
if __name__ == "__main__":
    train_model(device=torch.device("cpu"), batch_size=4, num_epochs=20)
