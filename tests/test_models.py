"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest


TEST_VEC_ZEROS = [[0,0],[0,0],[0,0]]

@pytest.mark.parametrize(
    "test, expected",
    [
     (TEST_VEC_ZEROS, [0,0]),
     ([[1, 2], [3, 4], [5, 6]], [3,4])
     ])
def test_daily_mean(test, expected):
    """test daily mean function on several input vectors"""
    from inflammation.models import daily_mean
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))



@pytest.mark.parametrize(
    "test, expected",
    [
     (TEST_VEC_ZEROS, [0,0]),
     ([[3,5],[5,7],[9,12]], [3,5]),
     ([[-3,-5],[6,-3],[-1,-3]], [-3,-5])
     ])
def test_daily_min(test, expected):
    """ trivial test for daily_min """
    from inflammation.models import daily_min
    npt.assert_array_equal(daily_min(np.array(test)), expected)

@pytest.mark.parametrize(
    "test, expected",
    [
     (TEST_VEC_ZEROS, [0,0]),
     ([[3,5],[5,7],[9,12]], [9,12]),
     ([[-3,-5],[6,-3],[-1,-3]], [6,-3])
     ])
def test_daily_max(test, expected):
    """ trivial test for daily max """
    from inflammation.models import daily_max
    npt.assert_array_equal(daily_max(np.array(test)), expected)

def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])



# TODO(lesson-robust) Implement tests for the other statistical functions
