"""
This module only job is to hold the list of patient data and
implement the `__len__` and `__getitem__` methods required by PyTorch.
"""

# src.data_preproc.dataset

from typing import List, Dict, Any
from torch.utils.data import Dataset


class PhysioNetDataset(Dataset):
    """
    A PyTorch Dataset to wrap the preprocessed PhysioNet data.
    Each item in the dataset is a dictionary as defined in your final spec.
    """

    def __init__(self, processed_data: List[Dict[str, Any]]) -> None:
        """
        Args:
            processed_data (List[Dict[str, Any]]): A list where each element
                is a dictionary containing a patient's full data.
        """
        self.data = processed_data

    def __len__(self) -> int:
        """Returns the number of patients in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a single patient's data dictionary.
        """
        return self.data[idx]
