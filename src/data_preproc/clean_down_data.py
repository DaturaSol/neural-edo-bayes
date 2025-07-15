"""
Module responsible for cleaning downloaded data from PhysioNet.
Formatting such to be ready for use in further processing.
"""

# src.data_preproc.clean_down_data

import re
import wfdb
import numpy as np
from pathlib import Path
from scipy.signal import resample
from biosppy.signals import tools
from typing import Dict, Any, Literal, Optional, Tuple


from src.data_preproc.config import PhysioNetConfig

CONFIG = PhysioNetConfig()
DATABASES = CONFIG.databases
BEAT_TYPE_MAP = CONFIG.beat_type_map
NUM_BEAT_CLASSES = CONFIG.num_beat_classes
PROCESSING_FS = CONFIG.processing_fs


def process_recording_single_worker(
    record_id: str, db_name: str, annot_ext: str, data_dir: Path
) -> Dict[str, Any]:
    """
    Processes a single patient's recording data and returns a dictionary containing
    structured information for downstream analysis.

    Returns
    ------
    dict
        Dictionary with the following keys:
        - id (str):
        - label (int):
        - label_name (str):
        - age (int):
        - sex (int):
        - times (np.ndarray, shape (n_beats,)): Array of beat times.
        - values (np.ndarray, shape (n_beats, 7)):
        `[R-R interval, 5 x One-Hot Beat Type, QRS Duration]`
        - masks (np.ndarray, shape (n_beats, 7)): Mask array indicating when beat types are valid.

    Notes
    -----
    - Beat types are one-hot encoded.
    - The first beat typically has a missing R-R interval (mask value 0).
    - All arrays are aligned by heartbeat event.
    -------
    Data Example
    -----
    >>> {
    ...   "id": "mitdb_100",
    ...   "label": 1,
    ...   "label_name": "Normal Sinus Rhythm",
    ...
    ...   # --- Covariante Data ---
    ...   "age": 69,   # Parsed as an integer
    ...   "sex": 0,  # Parsed as an integer('M' = 0 or 'F = 1')
    ...
    ...   # A 1D array of timestamps for each expert-annotated heartbeat.
    ...   "times": np.array([0.752, 1.516, 2.283, ...]),
    ...
    ...   # A 2D array of the feature values for each event.
    ...   # It has 7 columns: [R-R, 5 x One-Hot Beat Type, QRS Duration].
    ...   "values": np.array([
    ...       [ 0.000,  1, 0, 0, 0, 0,   0.088 ],
    ...       [ 0.764,  1, 0, 0, 0, 0,   0.096 ],
    ...       [ 0.767,  0, 1, 0, 0, 0,   0.120 ],
    ...       ...
    ...   ]),
    ...
    ...   # A 2D array of 0s and 1s, aligned with "values".
    ...   # It tells the model which values are real (1) and which are missing (0).
    ...   "masks": np.array([
    ...       # R-R is missing for the first beat
    ...       [ 0,   1, 1, 1, 1, 1,   1 ],
    ...       # All features are present for subsequent beats
    ...       [ 1,   1, 1, 1, 1, 1,   1 ],
    ...       [ 1,   1, 1, 1, 1, 1,   1 ],
    ...       ...
    ...   ])
    ... }
    """
    # --- Fetch the recording data ---
    signal, fields, ann = fetch_recording(record_id, annot_ext, data_dir)

    # --- Parse patient metadata ---
    age, sex = parse_age_sex(fields.get("comments", ""))
    # --- Filters signal, keeping samples consistent with processing frequency ---
    filtered_signal = filter_signal(signal, fields["fs"])
    # --- Scale annotation times and prepare values and masks ---
    ann_samples, ann_times, values, masks = scale_annot_times_get_values_mask(
        fields["fs"], ann
    )
    # --- Loops through annotations to fill values and masks ---
    previous_rpeak = None
    for idx, samp in enumerate(ann_samples):
        # R-R Interval
        t = ann_times[idx]
        if previous_rpeak is not None:  # I think setting this is faster.
            values[idx, 0] = t - previous_rpeak
            # No need to set mask for R-R interval, as it is always valid.
        else:
            masks[idx, 0] = 0.0  # First beat has no R-R interval
        previous_rpeak = t
        one_hot_vec = one_hot_beat_types(idx, ann)
        values[idx, 1:-1] = one_hot_vec  # Fill in the one-hot encoded beat types.
        values[idx, -1] = get_qrs_duration(
            filtered_signal, samp, fields["fs"]
        )  # QRS duration.
    label = next(db["label"] for db in DATABASES if db["name"] == db_name)
    label_name = next(
        db["label_name"] for db in DATABASES if db["name"] == db_name
    )  # Get the label name from the database metadata.
    return {
        "id": record_id,
        "label": label,
        "label_name": label_name,
        "age": age,
        "sex": sex,
        "times": ann_times,
        "values": values,
        "masks": masks,
    }


def fetch_recording(
    record_id: str, annot_ext: str, data_dir: Path
) -> Tuple[np.ndarray, Dict, wfdb.Annotation]:
    # --- Load the recording data ---
    record_path = str(data_dir / record_id)
    signal, fields = wfdb.rdsamp(record_path)
    ann = wfdb.rdann(record_path, annot_ext)
    assert isinstance(signal, np.ndarray), "Signal data should be a numpy array."
    signal = signal[:, 0]  # Use only the first channel if multi-channel
    return signal, fields, ann


def parse_age_sex(comments: str) -> Tuple[Optional[int], Optional[int]]:
    full_comment, age, sex = " ".join(comments).strip(), None, None
    try:
        age_match = re.search(r"\d+", full_comment)  # Age will always be a digit.
        if age_match:
            age = int(age_match.group(0))  # The age is the first digit group found.

        sex_match = re.search(
            r"\b(?:M|F)\b", full_comment, re.IGNORECASE
        )  # Sex is either M or F.
        if sex_match:
            sex = 0 if sex_match.group(0).upper() == "M" else 1  # Making it binary.
    except Exception as e:
        return None, None  # If parsing fails, return None for both
    return age, sex


def get_qrs_duration(filtered_signal: np.ndarray, rpeak: int, frequency: int) -> float:
    """
    The annotations give us the R peak, but we need to calculate the QRS duration.
    So we will use the filtered signal to find the QRS duration, on a small window around the R peak.
    """
    search_window = int(0.1 * frequency)  # 100 ms window around the R peak
    start_idx = max(0, rpeak - search_window)
    end_idx = min(len(filtered_signal), rpeak + search_window)
    q_point, s_point = start_idx, end_idx  # Default to the window edges
    # Loops backwards to find the Q point
    for i in range(rpeak, start_idx, -1):
        if (
            filtered_signal[i - 1] * filtered_signal[i] < 0
        ):  # Zero crossing indicates Q point. Hopefully...
            q_point = i
            break
    # Loops forwards to find the S point
    for i in range(rpeak, end_idx):
        if (
            filtered_signal[i] * filtered_signal[i + 1] < 0
        ):  # Zero crossing indicates S point. Hopefully...
            s_point = i
            break

    return (s_point - q_point) / frequency  # Return duration in seconds


def filter_signal(signal: np.ndarray, frequency: int) -> np.ndarray:
    """
    Filters a given signal using a Butterworth bandpass filter in the frequency range of 3-45 Hz.
    If the input signal's sampling frequency (`frequency`) differs from the processing frequency (`PROCESSING_FS`),
    the signal is resampled to match `PROCESSING_FS` before filtering.
    -------
    ---
    Parameters
    ----------
    - signal : `np.ndarray`
        The input signal array to be filtered.
    - frequency : `int`
        The sampling frequency of the input signal.
    -------
    Returns
    -------
    - filtered_signal : `np.ndarray`
        The filtered signal array.
    -------
    Notes
    -----
    The filtering is performed using `tools.filter_signal` with the following arguments:
    - signal: `np.ndarray`
        The signal to be filtered (resampled if necessary).
    - ftype: `str`
        The type of filter to use. Here, "butter" specifies a Butterworth filter.
    - band: `str`
        The filter band type. "bandpass" applies a bandpass filter.
    - order: `int`
        The order of the filter. Higher order means a steeper roll-off.
    - frequency: `list[int]`
        The cutoff frequencies for the bandpass filter, here [3, 45] Hz.
    - sampling_rate: `int`
        The sampling rate to use for filtering, set to `PROCESSING_FS`.
    """
    signal_to_filter = signal.copy()
    if frequency != PROCESSING_FS:
        signal_to_filter = resample(
            signal_to_filter, int(len(signal_to_filter) * PROCESSING_FS / frequency)
        )

    filtered_signal, _, _ = tools.filter_signal(
        signal=signal_to_filter,
        ftype="butter",
        band="bandpass",
        order=4,
        frequency=[3, 45],
        sampling_rate=PROCESSING_FS,
    )

    return filtered_signal


def scale_annot_times_get_values_mask(
    frequency: int, ann: wfdb.Annotation
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Scales annotation times to the processing frequency and prepares values and masks for each beat type.
    """
    ann_samples: np.ndarray = ann.sample  # Samples are already in the correct format
    ann_times = ann_samples / frequency  # Convert samples to seconds
    if frequency != PROCESSING_FS:
        ann_samples = np.round(ann_times * PROCESSING_FS / frequency, 0).astype(int)

    num_events = len(ann_times)
    num_features = 7  # R-R interval, 5 beat types, QRS duration
    values = np.zeros((num_events, num_features), dtype=np.float32)
    masks = np.ones_like(values, dtype=np.float32)

    return ann_samples, ann_times, values, masks


def one_hot_beat_types(idx: int, ann: wfdb.Annotation) -> np.ndarray:
    """
    One-hot encodes the beat types in the annotation and updates the values and masks arrays.
    """
    sym = ann.symbol[idx]  # type: ignore
    beat_class = BEAT_TYPE_MAP.get(sym, 4)  # 4 is for unclassifiable or paced beats.
    one_hot_vec = np.zeros(NUM_BEAT_CLASSES, dtype=np.float32)
    one_hot_vec[beat_class] = 1.0

    return one_hot_vec
