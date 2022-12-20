import plyformat
import numpy as np


def test_read_triangle():
    """Read a triangular face (ASCII format)."""
    v, f, e = plyformat.read_ply("data/triangle.ply")
    assert v.shape == (3,)
    assert len(v.dtype) == 3
    assert f.shape == (1,)
    assert len(f.dtype) == 1
    assert np.all(v["x"] == [0, 1.5, 0.75])
    assert np.all(v["y"] == [0, 0, 1])
    assert np.all(v["z"] == [0, 0, 0])
    assert f["vertex_index"][0] == [0, 1, 2]
    assert e == {}


def test_types():
    """Test all PLY types."""
    v, f, e = plyformat.read_ply("data/types.ply")
    assert tuple(v[0]) == (-1.25, -2.5, -3.75, 1, 2, 3, 4, 5, 6)
    assert tuple(f[0]) == ([0, 0, 0], [1, 2, 3, 4])
    assert e == {}


def test_read_tetrahedron():
    """Read a tetrahedron (binary format)."""
    v, f, e = plyformat.read_ply("data/tetrahedron.ply")
    assert v.shape == (4,)
    assert len(v.dtype) == 6
    assert f.shape == (4,)
    assert len(f.dtype) == 1
    assert e == {}
