import numpy as np
from numpy.lib import recfunctions
import struct
import collections


# Specifications from:
#
#   http://paulbourke.net/dataformats/ply/
#
# PLY files cointain named scene elements.  Files should include the
# "vertex" and the "face" elements, but other elements ("edge",
# "material", ...) may be also present.
#
# Each element is actually a list of items, each one with properties.
# Vertices should have at least the "x", "y" and "z" properties, and
# faces should have at least the "vertex_index" list property.
#
# This module uses numpy structured arrays to represent elements.

# TODO:
# - write binary
# - read ascii
# - read binary
# - test
# - setup


class PLYError(RuntimeError):
    pass


_TYPE_MAP = [
    (np.float64, b"double"),
    (np.float32, b"float"),
    (np.uint8, b"uchar"),
    (np.int8, b"char"),
    (np.uint16, b"ushort"),
    (np.int16, b"short"),
    (np.uint32, b"uint"),
    (np.int32, b"int"),
    (np.uint64, b"uint"),
    (np.int64, b"int")
]


def _ply_type(dtype):
    for dt, typename in _TYPE_MAP:
        if np.issubdtype(dtype, dt):
            return typename
    return None


def _list_types(arr):
    n = max((len(x) for x in arr), default=0)
    if n < 256:
        len_type = b"uchar"
    elif n < 65536:
        len_type = b"ushort"
    else:
        len_type = b"uint"
    if arr.shape[0] == 0:
        item_type = b"int"
    else:
        item_type = _ply_type(np.asarray(arr[0]).dtype)
        if item_type is None:
            raise PLYError("Invalid list item type")
    return len_type, item_type


def _write_properties(f, arr):
    for field in arr.dtype.fields.items():
        if np.issubdtype(field[1][0], np.object_):
            lt, it = _list_types(arr[field[0]])
            t = (lt, it, field[0].encode("ascii"))
            f.write(b"property list %s %s %s\n" % t)
        else:
            typename = _ply_type(field[1][0])
            if typename is None:
                raise PLYError(f"Invalid data type '{field[0]}'")
            f.write(b"property %s %s\n" % (typename, field[0].encode("ascii")))


def _write_ascii_element(f, arr):
    if arr.dtype.hasobject:
        # Lists need to be special cased.
        lists = set(i for i, f in enumerate(arr.dtype.fields.items())
                    if np.issubdtype(f[1][0], np.object_))
        n = arr.shape[0]
        for i in range(n):
            row = list(arr[i])
            data = []
            for j, x in enumerate(row):
                if j in lists:
                    # Lists are encoded as (length, item[0], item[1], ...).
                    data.append(str(len(x)))
                    data.extend(map(str, x))
                else:
                    # Regular elements are just added to the output.
                    data.append(str(x))
            f.write((" ".join(data)).encode("ascii"))
            f.write(b"\n")
    else:
        # Fast path for numerical-only elements.
        np.savetxt(f, arr, fmt="%g")


def _write_binary_element(f, data, big_endian):
    pass


def _write_ply_handle(f, elements, comments, binary, big_endian):
    f.write(b"ply\n")
    if ascii:
        f.write(b"format ascii 1.0\n")
    elif big_endian:
        f.write(b"format binary_big_endian 1.0\n")
    else:
        f.write(b"format binary_little_endian 1.0\n")
    for comment in comments.splitlines():
        f.write(b"comment " + comment.encode("ascii") + b"\n")
    for name, arr in elements.items():
        f.write(b"element %s %d\n" % (name.encode("ascii"), arr.shape[0]))
        _write_properties(f, arr)
    f.write(b"end_header\n")
    for arr in elements.values():
        if binary:
            _write_binary_element(f, arr, big_endian)
        else:
            _write_ascii_element(f, arr)


def write_ply(filename, vertices, faces, other_elements=None,
              comments="", binary=False, big_endian=False):
    """Write data to a PLY file."""
    vertices = np.asarray(vertices)
    if vertices.dtype.fields is None:
        vertices = recfunctions.unstructured_to_structured(
            vertices, names=["x", "y", "z"]
        )
    if not isinstance(faces, np.ndarray) or faces.dtype.fields is None:
        temp = np.empty(len(faces), dtype=[("vertex_index", "O")])
        for i, face in enumerate(faces):
            temp[i] = list(face)
        faces = temp
    else:
        faces = np.asarray(faces)
    elements = collections.OrderedDict()
    elements["vertex"] = vertices
    elements["face"] = faces
    if other_elements is not None:
        if "vertex" in other_elements or "face" in other_elements:
            raise PLYError("Invalid element name")
        elements.update(other_elements)
    if isinstance(filename, str):
        with open(filename, "wb") as f:
            _write_ply_handle(f, elements, comments, binary, big_endian)
    else:
        _write_ply_handle(filename, elements, comments, binary, big_endian)


def cube():
    v = [[x, y, z] for x in [0, 1] for y in [0, 1] for z in [0, 1]]
    f = [
        [0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
        [2, 3, 6, 7], [0, 2, 6, 4], [1, 3, 7, 5]
    ]
    return v, f


def rgb_cube():
    v = np.zeros((8,), dtype=[("x", "f8"), ("y", "f8"), ("z", "f8"),
                              ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"][4:] = 1
    v["y"][2::4] = 1
    v["y"][3::4] = 1
    v["z"][1::2] = 1
    v["red"] = 255
    v["green"] = 128
    f = [
        [0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
        [2, 3, 6, 7], [0, 2, 6, 4], [1, 3, 7, 5]
    ]
    return v, f


def _main():
    import io
    v, f = cube()
    buf = io.BytesIO()
    write_ply(buf, v, f, comments="test")
    print(buf.getvalue().decode("ascii"))


if __name__ == "__main__":
    _main()
