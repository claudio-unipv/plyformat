import plyformat
import numpy as np
import io
import pytest


@pytest.mark.parametrize("correct, error", [
    ("ply", "pl"),
    ("format", "Format"),
    ("ascii", "asci"),
    ("comment", "comments"),
    ("element", "property"),
    ("property", "prop"),
    ("float", "decimal"),
    ("int", "number"),
    ("list", "lost"),
    ("list", ""),
    ("end_header", "")
    ])
def test_header_error(correct, error):
    with open("data/triangle.ply") as f:
        ply = f.read()
    ply = ply.replace(correct, error)
    buf = io.BytesIO(ply.encode("ascii"))
    with pytest.raises(plyformat.PLYError):
        plyformat.read_ply(buf)


@pytest.mark.parametrize("correct, error", [
    (" 0 ", " 0 0 "),
    (" 0 ", " "),
    (" 0 ", " z "),
    ("\n", "\n1 2 3\n"),
    ("0 0 0\n", ""),
    ("3 0 1 2\n", "")
])
def test_ascii_data_error(correct, error):
    with open("data/triangle.ply") as f:
        ply = f.read()
    parts = ply.partition("end_header")
    ply2 = parts[0] + parts[1] + parts[2].replace(correct, error)
    buf = io.BytesIO(ply2.encode("ascii"))
    with pytest.raises(plyformat.PLYError):
        plyformat.read_ply(buf)


def test_binary_data_error():
    with open("data/tetrahedron.ply", "rb") as f:
        ply = f.read()
    start = ply.find(b"end_header\n") + len(b"end_header\n")
    end = len(ply)
    # for index in range(start, end, 7):
    index = start + 9
    ply2 = ply[:index] + ply[index + 25:]
    buf = io.BytesIO(ply2)
    with pytest.raises(plyformat.PLYError):
        plyformat.read_ply(buf)
    ply2 = ply + b"x"
    buf = io.BytesIO(ply2)
    with pytest.raises(plyformat.PLYError):
        plyformat.read_ply(buf)
