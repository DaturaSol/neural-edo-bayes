"""
Configuration module for the PhysioNet dataset.
This module defines the `PhysioNetConfig` class, which encapsulates configuration parameters
for working with various PhysioNet ECG databases. It includes paths for data storage,
database metadata, beat type mappings, and processing settings.
Classes:
    PhysioNetConfig: Holds configuration for dataset directories, database definitions,
                     beat type mappings, and processing parameters.
Attributes (PhysioNetConfig):
    version (str): Version of the configuration.
    database_dir (Path): Root directory for PhysioNet data.
    database_out_dir (Path): Directory for processed database outputs.
    final_data_dir (Path): Directory for final data splits.
    databases (list): List of dictionaries, each describing a PhysioNet database with
                      name, label, annotation extension, and record identifiers.
    beat_type_map (dict): Mapping of beat annotation symbols to integer class labels.
    num_beat_classes (int): Number of unique beat classes.
    processing_fs (int): Target sampling frequency for processing.
Example usage:
    config = PhysioNetConfig()
    print(config.database_dir)
    print(config.databases)
Configuration for physionet dataset, includes 
"""

# src.data_preproc.config

from pathlib import Path

class PhysioNetConfig:
    """
    Configuration class for PhysioNet ECG datasets preprocessing.
    Attributes:
        version (str): Version of the configuration.
        database_dir (Path): Directory where raw PhysioNet databases are stored.
        database_out_dir (Path): Directory for processed database outputs.
        final_data_dir (Path): Directory for final split data.
        databases (list): List of dictionaries, each containing metadata for a PhysioNet database:
            - name (str): Database name.
            - label (int): Integer label for the database.
            - label_name (str): Human-readable label name.
            - annot_ext (str): Annotation file extension.
            - records (list): List of record identifiers.
        beat_type_map (dict): Mapping of beat annotation symbols to integer class labels.
        num_beat_classes (int): Number of unique beat classes.
        processing_fs (int): Target sampling frequency for processing.
    Methods:
        __init__():
            Initializes the configuration with default paths, database metadata, beat type mapping, and processing parameters.
    """
    def __init__(self) -> None:
        self.version = "1.0.0"
        self.database_dir = Path.cwd() / ".data" / "physionet"
        self.database_out_dir = self.database_dir / "out"
        self.check_point_dir = self.database_dir / "model"
        self.final_data_dir = self.database_out_dir / "final_splits"
        # --- Initialize directories ---
        self.database_dir.mkdir(parents=True, exist_ok=True)
        self.database_out_dir.mkdir(parents=True, exist_ok=True)
        self.check_point_dir.mkdir(parents=True, exist_ok=True)
        self.final_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.databases = [
            {
                "name": "nsrdb",
                "label": 0,
                "label_name": "Normal Sinus Rhythm",
                "annot_ext": "atr",
                "records": [
                    "16265","16272","16273","16420","16483","16539","16773","16786",
                    "16795","17052","17453","18177","18184","19088","19090","19093",
                    "19140","19830",
                ],
            },
            {
                "name": "mitdb",
                "label": 1,
                "label_name": "Arrhythmia",
                "annot_ext": "atr",
                "records": [
                    "100","101","102","103","104","105","106","107","108","109","111",
                    "112","113","114","115","116","117","118","119","121","122","123",
                    "124","200","201","202","203","205","207","208","209","210","212",
                    "213","214","215","217","219","220","221","222","223","228","230",
                    "231","232","233","234",
                ],
            },
            {
                "name": "chfdb",
                "label_name": "Congestive Heart Failure",
                "label": 2,
                "annot_ext": "ecg",
                "records": [f"chf{str(i).zfill(2)}" for i in range(1, 16)],
            },
        ]
        
        self.beat_type_map = {
            "N": 0, "L": 0, "R": 0, "e": 0, "j": 0, # 0. Normal-like Beats
            "V": 1, "E": 1, # 1. Ventricular Ectopic Beats
            "A": 2, "a": 2, "J": 2, "S": 2, # 2. Supraventricular Ectopic Beats
            "F": 3, # 3. Fusion Beats
            "/": 4, "f": 4, "Q": 4, "?": 4, # 4. Unclassifiable or Paced Beats
        }
        
        self.num_beat_classes = len(set(self.beat_type_map.values()))
        
        self.processing_fs = 250 