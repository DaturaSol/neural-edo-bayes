"""
Responsible for running all the processes needed for a clean dataset.
This version is optimized for memory efficiency by processing and saving
one database at a time.
"""

# src.data_preproc.process_down_data

import os
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.data_preproc.config import PhysioNetConfig
from src.data_preproc.clean_down_data import process_recording_single_worker

CONFIG = PhysioNetConfig()
DATABSES = CONFIG.databases
DATABASE_DIR = CONFIG.database_dir
DATABASE_DIR_OUT = CONFIG.database_out_dir


def process_and_save_database(num_workes: int | None = None) -> None:
    """
    Processes each database in sequence, parallelizing the records within it.
    After each database is fully processed, its results are immediately saved
    to a .pt file to keep memory usage low.
    """
    if num_workes is None:
        num_workes = max(1, (os.cpu_count() or 1) // 2)

    print(f"Processing databases with {num_workes} workers...")

    # --- Loop through each database ---
    for db in DATABSES:
        db_name = db["name"]
        data_dir = DATABASE_DIR / db_name
        print(f"\nProcessing database: {db_name}")

        results, faliers = [], []

        with ProcessPoolExecutor(max_workers=num_workes) as executor:
            futures = {
                executor.submit(
                    process_recording_single_worker,
                    record,
                    db_name,
                    db["annot_ext"],
                    data_dir,
                ): record
                for record in db["records"]
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=db_name):
                record = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing record {db_name}/{record}: {e}")
                    faliers.append(record)
        if faliers:
            print(f"Finished processing {db_name}, failed {len(faliers)} records.")

        if results:
            try:
                out_path = DATABASE_DIR_OUT / f"{db_name}_processed.pt"
                torch.save(results, out_path)
                print(f"Saved processed data for {db_name} to {out_path}")
            except Exception as e:
                print(f"Error saving processed data for {db_name}: {e}")


# Run the processing function if this script is executed directly
if __name__ == "__main__":
    process_and_save_database()
