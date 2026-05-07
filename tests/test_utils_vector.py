"""Tests for utils_vector — pure-Python helpers that don't need API keys."""

from __future__ import annotations

import pytest

import utils_vector as uv


def _line(start, end, layer="L", length=None):
    """Helper: VectorLine with defaults for the optional rendering fields."""
    if length is None:
        length = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
    return uv.VectorLine(
        start=start,
        end=end,
        length_pts=length,
        color=None,
        width=1.0,
        dashes="",
        layer=layer,
    )


def test_vectorline_length():
    line = _line((0.0, 0.0), (3.0, 4.0))
    assert line.length_pts == pytest.approx(5.0)


def test_points_close_within_tolerance():
    assert uv.points_close((0.0, 0.0), (1.0, 1.0), tolerance=2.0) is True
    assert uv.points_close((0.0, 0.0), (5.0, 0.0), tolerance=2.0) is False


def test_calculate_total_length_aggregates_lines():
    lines = [
        _line((0.0, 0.0), (0.0, 10.0), layer="A"),
        _line((0.0, 0.0), (20.0, 0.0), layer="B"),
    ]
    result = uv.calculate_total_length(lines, scale_factor=30.0)
    # length in points: 10 + 20 = 30
    # unscaled inches: 30/72; scaled inches: that * 30; feet: scaled / 12
    assert result["segment_count"] == 2
    assert result["total_pts"] == pytest.approx(30.0)
    assert result["scale_factor"] == 30.0
    assert result["total_inches_unscaled"] == pytest.approx(30.0 / 72.0)
    assert result["total_inches_scaled"] == pytest.approx(30.0 / 72.0 * 30.0)
    assert result["total_feet"] == pytest.approx(30.0 / 72.0 * 30.0 / 12.0)


def test_group_connected_lines_finds_chains():
    """Two collinear segments sharing an endpoint should group together."""
    lines = [
        _line((0.0, 0.0), (10.0, 0.0)),
        _line((10.0, 0.0), (20.0, 0.0)),
        _line((50.0, 50.0), (60.0, 50.0)),
    ]
    groups = uv.group_connected_lines(lines, tolerance=1.0)
    # The two collinear ones form one group of 2; the disjoint one forms its own.
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 2]


def test_infer_scale_from_text_recognises_common_patterns():
    """Common architectural scale strings (1:100, 1/8" = 1', etc.) parse."""
    s1 = uv.infer_scale_from_text('1:100')
    assert s1 is not None and s1 > 0

    s2 = uv.infer_scale_from_text('SCALE: 1/8" = 1\'-0"')
    assert s2 is not None and s2 > 0


def test_infer_scale_from_text_returns_none_for_garbage():
    assert uv.infer_scale_from_text("no scale info here at all") is None
    assert uv.infer_scale_from_text("") is None


def test_distance_point_to_line_segment():
    # Distance from (5, 5) to segment (0,0)→(10,0) is 5.
    d = uv.distance_point_to_line_segment((5.0, 5.0), (0.0, 0.0), (10.0, 0.0))
    assert d == pytest.approx(5.0)

    # Distance from (3, 4) to segment (0,0)→(0,0) (degenerate) is 5 (point distance).
    d2 = uv.distance_point_to_line_segment((3.0, 4.0), (0.0, 0.0), (0.0, 0.0))
    assert d2 == pytest.approx(5.0)
