"""
Main training script for the GRU-ODE-Bayes model.

This script serves as the entry point for the training pipeline. It is based
on the original implementation by E. De Brouwer, available at:
https://github.com/edebrouwer/gru_ode_bayes

When executed, this script performs the following steps:
1. Prepares and processes the training data from its raw format.
2. Initializes and starts the model training process.

NOTE: Data preprocessing can be time-consuming. If the data has already
been processed, consider commenting out the `prepare_training_data()` call
to save time on subsequent runs.
"""

from torch import device

from src.data_preproc import prepare_training_data
from src.train_val.train_model import train_model


if __name__ == "__main__":
    # Step 1: Prepare and process the raw data.
    # This can be commented out if the data is already processed.
    prepare_training_data()

    # Step 2: Train the model.
    # A CPU device and a small batch size are used due to the high
    # computational overhead of the adjoint-based ODE solver, which was
    # a key finding of this project's analysis.
    train_model(device=device("cpu"), batch_size=1)
