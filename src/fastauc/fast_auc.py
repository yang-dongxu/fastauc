import ctypes
import os
from typing import Union
import numpy as np
from numpy.ctypeslib import ndpointer
import numpy as np
from pathlib import Path

class FastAuc:
    """A python wrapper class for a C++ library, used to load it once and make fast calls after.
    NB be aware of data types accepted, see method docstrings. 
    """

    def __init__(self):
        lib_path_pattern = Path(__file__).parent / "cpp_auc*.so"
        lib_path = list(lib_path_pattern.parent.glob(lib_path_pattern.name))
        if not lib_path:
            raise FileNotFoundError(f"The shared library file {lib_path_pattern} was not found.")
        if len(lib_path) > 1:
            raise FileExistsError("Multiple shared library files found: {}".format(lib_path))
        
        self._handle = ctypes.CDLL(lib_path[0])

        self._handle.cpp_auc_ext.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                             ndpointer(ctypes.c_bool, flags="C_CONTIGUOUS"),
                                             ctypes.c_size_t,
                                             ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                             ctypes.c_size_t]
        self._handle.cpp_auc_ext.restype = ctypes.c_float

        self._handle.cpp_aupr_ext.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                              ndpointer(ctypes.c_bool, flags="C_CONTIGUOUS"),
                                              ctypes.c_size_t,
                                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                              ctypes.c_size_t]
        self._handle.cpp_aupr_ext.restype = ctypes.c_float

    def roc_auc_score(self, y_true: np.array, y_score: np.array, sample_weight: np.array = None) -> float:
        """a method to calculate AUC via C++ lib.

        Args:
            y_true (np.array): 1D numpy array of dtype=np.bool8 as true labels.
            y_score (np.array): 1D numpy array of dtype=np.float32 as probability predictions.
            sample_weight (np.array): 1D numpy array as sample weights, optional.

        Returns:
            float: AUC score
        """
        y_true = np.asarray(y_true, dtype=np.bool8)
        y_score = np.asarray(y_score, dtype=np.float32)
        n = len(y_true)
        n_sample_weights = len(sample_weight) if sample_weight is not None else 0
        if sample_weight is None:
            sample_weight = np.array([], dtype=np.float32)
        result = self._handle.cpp_auc_ext(y_score, y_true, n, sample_weight, n_sample_weights)
        return result

    def pr_auc_score(self, y_true: np.array, y_score: np.array, sample_weight: np.array = None) -> float:
        """a method to calculate AUPR via C++ lib.

        Args:
            y_true (np.array): 1D numpy array of dtype=np.bool8 as true labels.
            y_score (np.array): 1D numpy array of dtype=np.float32 as probability predictions.
            sample_weight (np.array): 1D numpy array as sample weights, optional.

        Returns:
            float: AUPR score
        """
        y_true = np.asarray(y_true, dtype=np.bool8)
        y_score = np.asarray(y_score, dtype=np.float32)
        n = len(y_true)
        n_sample_weights = len(sample_weight) if sample_weight is not None else 0
        if sample_weight is None:
            sample_weight = np.array([], dtype=np.float32)
        result = self._handle.cpp_aupr_ext(y_score, y_true, n, sample_weight, n_sample_weights)
        return result

def fast_auc(y_true: np.array, y_score: np.array, sample_weight: np.array=None) -> Union[float, str]:
    """a function to calculate AUC via python.

    Args:
        y_true (np.array): 1D numpy array as true labels.
        y_score (np.array): 1D numpy array as probability predictions.
        sample_weight (np.array): 1D numpy array as sample weights, optional.

    Returns:
        float or str: AUC score or 'error' if imposiible to calculate
    """
    # binary clf curve
    y_true = (y_true == 1)

    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        sample_weight = sample_weight[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    if sample_weight is not None:
        tps = np.cumsum(y_true * sample_weight)[threshold_idxs]
        fps = np.cumsum((1 - y_true) * sample_weight)[threshold_idxs]
    else:
        tps = np.cumsum(y_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps

    # roc
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    if fps[-1] <= 0 or tps[-1] <= 0:
        return np.nan

    # auc
    direction = 1
    dx = np.diff(fps)
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            return 'error'

    area = direction * np.trapz(tps, fps) / (tps[-1] * fps[-1])

    return area

