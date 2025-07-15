"""
The training data varies in size greatly, so we need to chunk it in similar sizes.
Also includes steps as normalizing the data, splitting it into training and validation, test sets.
"""

# src.data_preproc.prepare_training_data

import torch
import joblib
import numpy as np
from typing import List, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.data_preproc.config import PhysioNetConfig

CONFIG = PhysioNetConfig()
DATABASE_OUT_DIR = CONFIG.database_out_dir
FINAL_DATA_DIR = CONFIG.final_data_dir
RANDOM_STATE = 37  # For reproducibility
CHUNK_SIZE = 512  # Size of each chunk for training data
CHUNK_OVERLAP = 128  # Overlap between chunks


def chunk_patient_data(
    patient_data: Dict[str, Any],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Splits a single patient's data into multiple smaller, overlapping chunks.
    """
    times = patient_data["times"]
    values = patient_data["values"]
    masks = patient_data["masks"]

    num_events = len(times)
    if num_events <= chunk_size:
        return [patient_data]

    chunks = []
    start_idx = 0
    while start_idx < num_events:
        end_idx = min(start_idx + chunk_size, num_events)
        # Chunk of static data.
        chunk = {
            "id": f"{patient_data['id']}_chunk{len(chunks)}",
            "label": patient_data["label"],
            "label_name": patient_data["label_name"],
            "age": patient_data["age"],
            "sex": patient_data["sex"],
        }

        chunk_times = times[start_idx:end_idx]
        # NOTE: We need to make each chunk and idependent time series.
        chunk["times"] = chunk_times - chunk_times[0]  # Normalize time to start at 0
        chunk["values"] = values[start_idx:end_idx].copy()
        chunk["masks"] = masks[start_idx:end_idx].copy()
        chunks.append(chunk)

        # Move the start index forward by chunk size minus overlap
        start_idx += chunk_size - chunk_overlap
        if end_idx == num_events:
            break

    return chunks


def apply_normalization(
    dataset_split: List[Dict[str, Any]],
    feature_scaller: StandardScaler,
    age_scaller: StandardScaler,
) -> List[Dict[str, Any]]:
    normalized_split = []
    for patient_data in dataset_split:
        if patient_data["age"] is None:
            patient_data["age"] = age_scaller.mean_[0]  # type: ignore # Use mean age if None
        patient_norm = patient_data.copy()
        patient_norm["values"] = feature_scaller.transform(patient_data["values"])
        patient_norm["age"] = age_scaller.transform([[patient_data["age"]]]).reshape(
            1, -1
        )[0]
        normalized_split.append(patient_norm)

    return normalized_split


def save_split_norm_data() -> None:
    """
    Processes the downloaded PhysioNet data, normalizes it, splits it into training,
    validation, and test sets, and saves the final dataset splits.
    """
    print("Loading processed data...")
    all_patients = []
    for pt_file in DATABASE_OUT_DIR.glob("*.pt"):
        patient_data = torch.load(pt_file, weights_only=False)
        all_patients.extend(patient_data)
    print(f"Loaded {len(all_patients)} patients.")

    # --- Stratify split by label ---
    patient_labels = [p["label"] for p in all_patients]
    train_data, test_data = train_test_split(
        all_patients,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=patient_labels,
    )
    test_labels = [p["label"] for p in test_data]
    val_data, test_data = train_test_split(
        test_data,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=test_labels,
    )
    print(
        f"Split into {len(train_data)} train, {len(val_data)} val, {len(test_data)} test patients."
    )

    # --- Computes Normalization parameters ---
    print("Computing normalization parameters...")
    all_train_values = np.concatenate([p["values"] for p in train_data], axis=0)
    feature_scaler = StandardScaler()
    feature_scaler.fit(all_train_values)
    # NOTE: One-Hot features do not need normalization.
    feature_scaler.mean_[1:6] = 0.0  # type: ignore
    feature_scaler.scale_[1:6] = 1.0  # type: ignore
    all_train_ages = np.array([p["age"] for p in train_data]).reshape(-1, 1)
    age_scaler = StandardScaler()
    age_scaler.fit(all_train_ages)
    print("Normalization parameters computed.")

    # --- Save scalers ---
    print("Saving normalization parameters...")
    scalers = {"features": feature_scaler, "age": age_scaler}
    scalers_path = FINAL_DATA_DIR / "scalers.gz"
    joblib.dump(scalers, scalers_path)
    print(f"Saved scalers to {scalers_path}.")

    # --- Normalize splits ---
    print("Normalizing dataset splits...")
    train_data_norm = apply_normalization(train_data, feature_scaler, age_scaler)
    val_data_norm = apply_normalization(val_data, feature_scaler, age_scaler)
    test_data_norm = apply_normalization(test_data, feature_scaler, age_scaler)

    # --- Chunk the normalized data ---
    print(
        f"Chunking normalized data with chunk size {CHUNK_SIZE} and overlap {CHUNK_OVERLAP}..."
    )
    train_chunks = [
        chunk
        for p in train_data_norm
        for chunk in chunk_patient_data(p, CHUNK_SIZE, CHUNK_OVERLAP)
    ]
    val_chunks = [
        chunk
        for p in val_data_norm
        for chunk in chunk_patient_data(p, CHUNK_SIZE, CHUNK_OVERLAP)
    ]
    test_chunks = [
        chunk
        for p in test_data_norm
        for chunk in chunk_patient_data(p, CHUNK_SIZE, CHUNK_OVERLAP)
    ]
    print(
        f"Created {len(train_chunks)} train, {len(val_chunks)} val, {len(test_chunks)} test chunks."
    )

    # --- Save the final splits ---
    print("Saving final dataset splits...")
    torch.save(train_chunks, FINAL_DATA_DIR / "train_chunks.pt")
    torch.save(val_chunks, FINAL_DATA_DIR / "val_chunks.pt")
    torch.save(test_chunks, FINAL_DATA_DIR / "test_chunks.pt")
    print(f"Final dataset splits saved at '{FINAL_DATA_DIR}'.")
    

# run this file to process the data and save it in a clean format.
if __name__ == "__main__":
    save_split_norm_data()