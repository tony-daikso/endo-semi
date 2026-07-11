"""§7 FP/FN flagging and linear interpolation."""
import numpy as np

from endosemi.temporal import flag_fp_fn, interpolate_box


def test_isolated_track_flagged_fp():
    fp, fn = flag_fp_fn(tracks_prev={1}, tracks_cur={1, 2}, tracks_next={1})
    assert fp == {2}          # track 2 appears only in cur -> spurious
    assert fn == set()


def test_gap_flagged_fn():
    fp, fn = flag_fp_fn(tracks_prev={1}, tracks_cur=set(), tracks_next={1})
    assert fn == {1}          # track 1 in prev and next but missing in cur
    assert fp == set()


def test_continuous_track_not_flagged():
    fp, fn = flag_fp_fn(tracks_prev={1}, tracks_cur={1}, tracks_next={1})
    assert fp == set() and fn == set()


def test_linear_interpolation_midpoint():
    prev = [0, 0, 10, 10]
    nxt = [10, 10, 20, 20]
    mid = interpolate_box(prev, nxt, method="linear")
    np.testing.assert_allclose(mid, [5, 5, 15, 15])
