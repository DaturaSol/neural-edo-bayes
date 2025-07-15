"""
Module for data preprocessing tasks.
Includes downloading PhysioNet datasets and processing them into a clean format.
"""

# src.data_preproc

from src.data_preproc.config import PhysioNetConfig
from src.data_preproc.dataset import PhysioNetDataset
from src.data_preproc.collators import collate_function
from src.data_preproc.download_db import download_datasets
from src.data_preproc.process_down_data import process_and_save_database
from src.data_preproc.prepare_training_data import save_split_norm_data


def prepare_training_data() -> None:
    download_datasets()
    process_and_save_database()
    save_split_norm_data()
    print("Data preparation complete.")


# Run this file to process the data and save it in a clean format.
# Ready to be used in the training pipeline.
if __name__ == "__main__":
    prepare_training_data()
