"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains
inflammation data for a single patient taken over a number of days
and each column represents a single day across all patients.
"""

import numpy as np


def load_csv(filename: str) -> np.ndarray:
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data: np.ndarray) -> np.ndarray:
    """Calculate the daily mean of a 2D inflammation data array.

        Simply calles underlying numpy function
        :param data the array to find the daily mean of (2D numpy array, daily data row-wise)
        :returns array of mean values with daily means
    """
    return np.mean(data, axis=0)


def daily_max(data: np.ndarray) -> np.ndarray:
    """Calculate the daily max of a 2D inflammation data array.

        Simply calles underlying numpy function
        :param data the data to find the daily mean of (2D numpy array, day data row-wise)
        :returns array of mean values with daily maxima
    """
    return np.max(data, axis=0)


def daily_min(data: np.ndarray) -> np.ndarray:
    """Calculate the daily min of a 2D inflammation data array.

        Simply calles underlying numpy function
        :param data the data to find the daily mean of (2D numpy array, day data row-wise)
        :returns array of mean values with daily minima
    """
    return np.min(data, axis=0)


# TODO(lesson-design) Add Patient class
# TODO(lesson-design) Implement data persistence
# TODO(lesson-design) Add Doctor class
