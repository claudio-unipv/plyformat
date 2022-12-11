import numpy as np
from numpy.lib import recfunctions
import struct
import contextlib
import collections


__doc__ = """Read/write PLY files."""


__all__ = [
    "write_ply"
]


# Format specifications from:
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
# Field names and types corresponds to names and types of properties.
# An array with a field with object type is supposed to contain lists,
# and corresponds to a property list.
#
# As an exception, the "vertex" element can be represented as a Nx3
# array with the default properties named "x", "y" and "z".
#
# A second exception is the "face" element, that can be represented as
# a list of lists, and that will interpreted as having the single list
# property "vertex_index".


# TODO:
# - read ascii
# - read binary
# - test
# - setup


class PLYError(RuntimeError):
    """Error reading or writing a PLY file."""
    pass


# Numpy to PLY data types.
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


# PLY data type to struct format characters.
_STRUCT_MAP = {
    b"double": "d",
    b"float": "f",
    b"uchar": "B",
    b"char": "b",
    b"ushort": "H",
    b"short": "h",
    b"uint": "I",
    b"int": "i"
}


def _ply_type(dtype):
    """Return the corresponding PLY type, or None."""
    for dt, typename in _TYPE_MAP:
        if np.issubdtype(dtype, dt):
            return typename
    return None


def _list_types(arr):
    """Determine the types for the length and the items of a property list."""
    # The types for the length is determined on the basis of the longest list.
    n = max((len(x) for x in arr), default=0)
    if n < 256:
        len_type = b"uchar"
    elif n < 65536:
        len_type = b"ushort"
    else:
        len_type = b"uint"
    # The type for the items is determined by inspecting the first
    # list ("int" is taken as default).
    if arr.shape[0] == 0:
        item_type = b"int"
    else:
        item_type = _ply_type(np.asarray(arr[0]).dtype)
        if item_type is None:
            raise PLYError("Invalid list item type")
    return len_type, item_type


def _write_properties(f, arr):
    """Write to f the property declarations of arr."""
    for field in arr.dtype.fields.items():
        if np.issubdtype(field[1][0], np.object_):
            # Object fields are intepreted as property lists.
            lt, it = _list_types(arr[field[0]])
            t = (lt, it, field[0].encode("ascii"))
            f.write(b"property list %s %s %s\n" % t)
        else:
            # Numeric fields are intepreted as scalar properties.
            typename = _ply_type(field[1][0])
            if typename is None:
                raise PLYError(f"Invalid data type '{field[0]}'")
            f.write(b"property %s %s\n" % (typename, field[0].encode("ascii")))


def _write_ascii_element(f, arr):
    """Write the content of arr to f using the ascii format."""
    if arr.dtype.hasobject:
        # Lists need to be special cased.
        islist = [np.issubdtype(f[0], np.object_) for f in arr.dtype.fields.values()]
        for i in range(arr.shape[0]):
            data = []
            for x, xlist in zip(arr[i], islist):
                if xlist:
                    # Lists are encoded as (length, item[0], item[1], ...).
                    data.append(str(len(x)))
                    data.extend(map(str, x))
                else:
                    # Regular elements are just added to the output.
                    data.append(np.str_(x))
            f.write((" ".join(data)).encode("ascii"))
            f.write(b"\n")
    else:
        # Fast path for numerical-only elements.
        np.savetxt(f, arr, fmt="%g")


def _write_binary_element(f, arr, big_endian):
    """Write the content of arr to f using the binary format."""
    # Build a format string for the fields in arr.
    fmt = [(">" if big_endian else "<")]
    islist = []
    for field in arr.dtype.fields.items():
        if np.issubdtype(field[1][0], np.object_):
            # Object fields are intepreted as property lists, and
            # requires the output of the length and the items.
            lt, it = _list_types(arr[field[0]])
            fmt.append(_STRUCT_MAP[lt])
            fmt.append(" {:d}" + _STRUCT_MAP[it])
            islist.append(True)
        else:
            fmt.append(_STRUCT_MAP[_ply_type(field[1][0])])
            islist.append(False)
    fmt = "".join(fmt)
    # Write the data.
    if arr.dtype.hasobject:
        for row in arr:
            ls = []
            data = []
            for x, xlist in zip(row, islist):
                if xlist:
                    ls.append(len(x))
                    data.append(len(x))
                    data.extend(x)
                else:
                    data.append(x)
            f.write(struct.pack(fmt.format(*ls), *data))
    else:
        # For numeric-only elements the procedure is easier, and just
        # requires to write each composite item.
        for row in arr:
            f.write(struct.pack(fmt, *row))


def _write_ply_handle(f, elements, comments, binary, big_endian):
    """Write elements to the file-like object f."""
    # Write the header.
    f.write(b"ply\n")
    if not binary:
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
    # Write the data.
    for arr in elements.values():
        if binary:
            _write_binary_element(f, arr, big_endian)
        else:
            _write_ascii_element(f, arr)


def write_ply(filename, vertices, faces, other_elements=None,
              comments="", binary=False, big_endian=False):
    """Write data to a PLY file.

    Args:
        filename (str or file): Name of the file, or or file-like object.
        vertices (structured array): vertex data.
        faces (structured array): face data.
        other_elements (dict): named optional elements.
        comments (str): comments.
        binary (bool): use binary instead of ascii format.
        big_endial (bool): big instead of little endian encoding.

    Returns:
        None

    vertices, faces and the values in other_elements must be
    structured arrays, with named fields.  List properties are
    represented by fields of object type.

    If vertices is given as a Nx3 array, it is converted to a
    structured array with fields 'x', 'y' and 'z'.

    If faces is given as a list of lists, it is converted to a
    structured array with a single field of object type named
    'vertex_index'.

    other_elements may provide extra elements as a dictionary with
    names as keys and structured arrays as values.

    """
    # Convert vertices to a structured array, if needed.
    vertices = np.asarray(vertices)
    if vertices.dtype.names is None:
        # Names is None for regular unstructured arrays.
        vertices = recfunctions.unstructured_to_structured(
            vertices, names=["x", "y", "z"]
        )
    # Convert faces to a structured array, if needed.
    if not isinstance(faces, np.ndarray) or faces.dtype.names is None:
        # If not an array or not a structured one.
        lst = list(map(list, faces))
        faces = np.empty(len(lst), dtype=[("vertex_index", "O")])
        faces["vertex_index"] = lst
    # Build an ordered dict with all the elements.
    elements = collections.OrderedDict()
    elements["vertex"] = vertices
    elements["face"] = faces
    if other_elements is not None:
        if "vertex" in other_elements or "face" in other_elements:
            raise PLYError("Invalid element name")
        elements.update(other_elements)
    # Write the data.
    with contextlib.ExitStack() as stack:
        if isinstance(filename, str):
            f = stack.push(open(filename, "wb"))
        else:
            # If filename it's not a sting it's assumed to be a
            # file-like object.
            f = filename
        _write_ply_handle(f, elements, comments, binary, big_endian)


def cube():
    v = [[x, y, z] for x in [0.0, 1] for y in [0, 1] for z in [0, 1]]
    f = [
        [0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
        [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]
    ]
    return v, f


def prism(n):
    import math
    v = [[0, 1, 0], [0, -1, 0]]
    for i in range(n):
        a = 2 * math.pi * i / n
        v.append([math.cos(a), 1, math.sin(a)])
        v.append([math.cos(a), -1, math.sin(a)])
    f = []
    for i in range(n):
        a, b, c, d = [2 + j % (2 * n) for j in range(2 * i, 2 * i + 4)]
        f.append([0, c, a])
        f.append([1, b, d])
        f.append([a, c, d, b])
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
        [0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
        [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]
    ]
    return v, f


def _main():
    import io
    v, f = prism(10)
    buf = io.BytesIO()
    write_ply(buf, v, f, comments="test", binary=False)
    write_ply("a.ply", v, f, comments="test", binary=True)
    print(buf.getvalue().decode("ascii"), end="")


if __name__ == "__main__":
    _main()
