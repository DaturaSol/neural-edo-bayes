"""
Dowloads Physionet files from source and stores them in the specified directory.
"""

# src.data_preproc.download_db

import requests
from tqdm import tqdm
from pathlib import Path

from src.data_preproc.config import PhysioNetConfig

CONFIG = PhysioNetConfig()
DATABASE_DIR = CONFIG.database_dir
DATABASES = CONFIG.databases
VERSION = CONFIG.version


def download_datasets() -> None:
    print(
        f"Downloading PhysioNet datasets version {VERSION}...\nIn directory: {DATABASE_DIR}\n"
    )
    for db in DATABASES:
        db_dir = DATABASE_DIR / db["name"]
        db_dir.mkdir(parents=True, exist_ok=True)
        print("\nDownloading database:", db["name"])
        for record in db["records"]:
            for ext in ("dat", "hea", db["annot_ext"]):
                file_name = f"{record}.{ext}"
                url = f"https://physionet.org/files/{db['name']}/{VERSION}/{file_name}"
                dest = db_dir / file_name
                if dest.exists():
                    print(f"File {file_name} already exists, skipping download.")
                    continue
                try:
                    download_file(url, dest)
                except requests.HTTPError as e:
                    print(f"Failed to download {file_name}: {e}")
                except Exception as e:
                    print(f"An error occurred while downloading {file_name}: {e}")
    print("\nAll downloads completed.")


def download_file(url: str, dest: Path) -> None:
    """
    Downloads a file from the given URL to the specified destination path.
    """
    print(f"Downloading {url} to {dest}...")
    response = requests.get(url + "?download", stream=True)
    response.raise_for_status()  # Raise an error for bad responses
    total_size = int(response.headers.get("content-length", 0))

    with (
        open(dest, "wb") as file,
        tqdm(
            desc=dest.name,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in response.iter_content(chunk_size=1024):
            if not data:
                continue
            file.write(data)
            bar.update(len(data))


# Run the download function if this script is executed directly
if __name__ == "__main__":
    download_datasets()
