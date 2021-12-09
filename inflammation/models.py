"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains
inflammation data for a single patient taken over a number of days
and each column represents a single day across all patients.
"""

from dataclasses import dataclass, field
import numpy as np
from typing import List, Dict, Any
from collections import namedtuple

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

def patient_normalise(data: np.ndarray) -> np.ndarray:
    """Normalise patient data from a 2D inflammation data array."""
    patmax = np.max(data, axis=1)
    return data / patmax[:, np.newaxis]


# TODO(lesson-design) Add Patient class


Observation = namedtuple("Observation", ["day", "value"])
Person = namedtuple("Person", ["name"])

@dataclass
class Patient:
    name: str
    observations: List[Observation] = field(default_factory=list, init=True)
    
    def add_observation(self, value, day=None) -> Observation:
        try:
            lday = day if day is not None else self.observations[-1]["day"] +1
        except IndexError:
            lday = 0

        new_obs = Observation(lday, value)
        self.observations.append(new_obs)
        new_obs = Observation(lday, value)
        return new_obs
        
    @property
    def last_observation(self) -> int:
        return self.observations[-1]
    
    
    def __len__(self) -> int:
        return len(self.observations)
    
    def __repr__(self) -> str:
        return "Patient: (%s)" % self.name
    
    def __str__(self) -> str:
        return self.name


@dataclass
class Doctor:
    name: str
    patients : List[Patient] = field(default_factory=list, init=False)
    


# TODO(lesson-design) Implement data persistence
# TODO(lesson-design) Add Doctor class
