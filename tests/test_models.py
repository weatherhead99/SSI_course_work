"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_dailiy_min():
    """ trivial test for daily_min """
    from inflammation.models import daily_min
    test_input = np.array([[1,1],
                           [5,7],
                           [9,2]])
    test_result = np.array([1,1])
    npt.assert_array_equal(daily_min(test_input), test_result)

def test_daily_max():
    """ trivial test for daily max """
    from inflammation.models import daily_max
    test_input = np.array([[1,1],
                           [5,7],
                           [9,2]])
    test_result = np.array([9,7])
    npt.assert_array_equal(daily_max(test_input), test_result)

def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])



# TODO(lesson-robust) Implement tests for the other statistical functions
