"""
Collators for preparing training data.
This module is responsible for outputing data in pytorch tensor compatible with the training pipeline.
The idea with this colator is to send different patient data in to a intial condition that our ODE solver can use.
"""

# src.data_preproc.collators

import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List


def collate_function(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collates a batch of patient data into a single dictionary, ready to be used in the forward pass of a model.
    -----
    ----
    Return a dictionary as follows:
    -----
    >>> {
    ...     "cov": covariates,
    ...     "X": X,
    ...     "M": M,
    ...     "times": times,
    ...     "time_ptr": time_ptr,
    ...     "obs_idx": obs_idx,
    ...     "labels": labels,
    ...     "T": T_batch,
    ... }

    """
    # --- Extract patient IDs, ages, and sexes ---
    patient_ids = [patient["id"] for patient in batch]
    id_to_batch_idx = {pid: idx for idx, pid in enumerate(patient_ids)}

    ages = torch.tensor(
        [patient["age"][0] for patient in batch], dtype=torch.float32
    )  #  Shape: [batch_size]

    sexes = torch.tensor([patient["sex"] for patient in batch], dtype=torch.float32)
    sex_one_hot = torch.nn.functional.one_hot(
        sexes.long(), num_classes=2
    ).float()  # Shape: [batch_size, 2]

    covariates = torch.cat([ages.unsqueeze(1), sex_one_hot], dim=1)

    # --- Prepare a DataFrame to hold all events ---
    # This will help us to create a time series representation of the data.
    all_events_df = []
    for p_data in batch:
        num_events = len(p_data["times"])
        df_dict = {
            "Time": p_data["times"],
            "PatientID": [p_data["id"]] * num_events,
        }
        # --- Add values and masks to the DataFrame ---
        for i in range(p_data["values"].shape[1]):
            df_dict[f"Value_{i}"] = p_data["values"][:, i]
            df_dict[f"Mask_{i}"] = p_data["masks"][:, i]
        # --- Convert to DataFrame and append to the list ---
        df = pd.DataFrame(df_dict)
        all_events_df.append(df)
    # --- Concatenate all DataFrames into a single DataFrame ---
    # This will allow us to handle all events in a single batch.
    batch_df = pd.concat(all_events_df, ignore_index=True)

    # Convert Time to float32 BEFORE finding unique values.
    batch_df["Time"] = batch_df["Time"].astype(np.float32)
    # Drop duplicates based on the float32 time representation.
    batch_df.drop_duplicates(subset=["Time", "PatientID"], inplace=True)
    # Sort deduplicated DataFrame by Time.
    batch_df.sort_values(by="Time", inplace=True, kind="mergesort")
    # It is now guaranteed to be sorted with no duplicates.
    times = torch.from_numpy(np.unique(batch_df["Time"].to_numpy()))
    if len(times) > 1:
        assert torch.all(
            torch.diff(times) > 0
        ), "Times tensor is not strictly increasing after cleaning!"

    # NOTE: LEGACY, we dont even use those on the main foward method anymore...
    time_counts = batch_df.groupby(
        "Time", sort=False
    ).size()  # This tells us how many observations happened at each unique timestamp.
    time_ptr = torch.cat(
        [
            torch.tensor([0]),  # We also need the start indices, so we concatenate a 0.
            torch.from_numpy(time_counts.cumsum().to_numpy()).to(
                torch.long
            ),  # CUM SUM --- Cumulative Summation,
            # is a function that calculates the cumulative sum along a given axis (axis 0 in this case).
            # For an input [a, b, c], the cumulative sum is [a, a+b, a+b+c].
            # Apply it to our `time_counts` array represents the end index (exclusive)
            # for the observations corresponding to each unique time.
        ]
    )  # It is a map where for any unique time times[i], the corresponding observations
    # in X and M are located from index time_ptr[i] up to (but not including) index time_ptr[i+1]
    assert (
        len(time_ptr) == len(times) + 1
    ), f"Mismatch: len(time_ptr)={len(time_ptr)}, len(times)={len(times)}"

    # Create the observation-to-time mapping
    time_to_idx_map = {t.item(): i for i, t in enumerate(times)}
    obs_to_time_idx = torch.tensor(
        batch_df["Time"].map(time_to_idx_map).to_numpy(), dtype=torch.long
    )

    # Extract all other tensors from the sorted DataFrame.
    value_cols = [col for col in batch_df.columns if col.startswith("Value_")]
    mask_cols = [col for col in batch_df.columns if col.startswith("Mask_")]

    X = torch.from_numpy(batch_df[value_cols].values).to(torch.float32)
    M = torch.from_numpy(batch_df[mask_cols].values).to(torch.float32)

    obs_idx = torch.tensor(
        [id_to_batch_idx[pid] for pid in batch_df["PatientID"]],
        dtype=torch.long,
    )

    labels = torch.tensor([p["label"] for p in batch], dtype=torch.long)
    T_batch = times.max() if len(times) > 0 else torch.tensor(0.0)

    return {
        "cov": covariates,
        "X": X,
        "M": M,
        "times": times,
        "time_ptr": time_ptr,
        "obs_idx": obs_idx,
        "labels": labels,
        "T": T_batch,
        "obs_to_time_idx": obs_to_time_idx,
    }
