import plyformat
import numpy as np
import io
import pytest


@pytest.mark.parametrize("binary,big_endian", [(False, False), (True, False), (True, True)])
def test_write(binary, big_endian):
    """Write/read (ASCII format)."""
    v = [[x, y, z] for x in [0.0, 1] for y in [0, 1] for z in [0, 1]]
    f = [
        [0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
        [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]
    ]
    edata = np.empty(10, dtype=[("property", "i")])
    edata["property"] = np.arange(10)
    e = {"extra": edata}
    buf = io.BytesIO()
    plyformat.write_ply(buf, v, f, e, comments="test", binary=binary, big_endian=big_endian)
    buf.seek(0)
    v2, f2, e = plyformat.read_ply(buf)
    assert [x[0] for x in v] == list(v2["x"])
    assert [x[1] for x in v] == list(v2["y"])
    assert [x[2] for x in v] == list(v2["z"])
    assert f == list(f2["vertex_index"])
    assert np.all(e["extra"] == edata)
