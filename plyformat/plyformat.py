import numpy as np
from numpy.lib import recfunctions
import struct
import contextlib
import collections


__doc__ = """Read/write PLY files."""
__all__ = ["PLYError", "read_ply", "write_ply"]


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
# - test
# - setup
# - catch errors due to ill-formed input files


class PLYError(RuntimeError):
    """Error reading or writing a PLY file."""
    pass


def read_ply(filename):
    """Read data from a PLY file.

    Args:
        filename (str or file): Name of the file, or or file-like object.

    Returns:
        vertices (structured array): vertex data.
        faces (structured array): face data.
        other_elements (dict): named optional elements.
    """
    with contextlib.ExitStack() as stack:
        if isinstance(filename, str):
            f = stack.push(open(filename, "rb"))
        else:
            # If filename it's not a sting it's assumed to be a
            # file-like object.
            f = filename
        elements = _read_ply_handle(f)
    # Rearrange the elements.
    if "vertex" in elements:
        vertex = elements.pop("vertex")
    else:
        vertex = np.empty((0,), dtype=[("x", "f"), ("y", "f"), ("z", "f")])
    if "face" in elements:
        face = elements.pop("face")
    else:
        face = np.empty((0,), dtype=[("vertex_index", "O")])
    return vertex, face, elements


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


# Numpy to PLY data types.
_NP2PLY_MAP = [
    (np.float64, b"double"),
    (np.float32, b"float"),
    (np.uint8, b"uchar"),
    (np.int8, b"char"),
    (np.uint16, b"ushort"),
    (np.int16, b"short"),
    (np.uint32, b"uint"),
    (np.int32, b"int")
]


# PLY data type to struct format characters.
_PLY2STRUCT_MAP = {
    b"double": "d",
    b"float": "f",
    b"uchar": "B",
    b"char": "b",
    b"ushort": "H",
    b"short": "h",
    b"uint": "I",
    b"int": "i"
}


# PLY data type to struct format characters.
_NP2STRUCT_MAP = {
    np.float64: "d",
    np.float32: "f",
    np.uint8: "B",
    np.int8: "b",
    np.uint16: "H",
    np.int16: "h",
    np.uint32: "I",
    np.int32: "i"
}


def _ply_type(dtype):
    """Return the corresponding PLY type, or None."""
    if np.issubdtype(dtype, np.int64):
        dtype = np.int32
    elif np.issubdtype(dtype, np.uint64):
        dtype = np.uint32
    for dt, typename in _NP2PLY_MAP:
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
            fmt.append(_PLY2STRUCT_MAP[lt])
            fmt.append(" {:d}" + _PLY2STRUCT_MAP[it])
            islist.append(True)
        else:
            fmt.append(_PLY2STRUCT_MAP[_ply_type(field[1][0])])
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


def _read_header_line(f):
    """Read a line from the header, ignoring comments."""
    while True:
        line = f.readline()
        if line is None:
            raise PLYError("Invalid PLY file (unexpected EOF in header)")
        line = line.strip()
        if len(line) > 0 and not line.startswith(b"comment "):
            return line


def _read_header_intro(f):
    line = _read_header_line(f)
    if line != b"ply":
        raise PLYError(f"Invalid PLY file (unexpected magic number '{line.hex()}')")
    line = _read_header_line(f)
    fmt = line.split()
    if len(fmt) != 3 or fmt[0] != b"format":
        line = line.decode("ascii")
        raise PLYError(f"Invalid PLY file (invalid format '{line}')")
    if fmt[1] == b"ascii":
        return False, True
    elif fmt[1] == b"binary_little_endian":
        return True, False
    elif fmt[1] == b"binary_big_endian":
        return True, True
    else:
        line = fmt[1].decode("ascii")
        raise PLYError(f"Invalid PLY file (invalid format '{line}')")


def _isvalidid(b):
    return b.decode("ascii").isidentifier()


def _read_header_definitions(f):
    """Read the definition of the elements."""
    name = None
    type_ = None
    size = None
    elements = collections.OrderedDict()
    list_types = {}
    types = {ply: dt for dt, ply in _NP2PLY_MAP}
    while True:
        line = _read_header_line(f)
        tokens = line.split()
        if tokens[0] == b"property":
            if type_ is None:
                raise PLYError("Invalid PLY file (unexpected property)")
            elif len(tokens) == 3 and tokens[1] in types and _isvalidid(tokens[2]):
                # Store a scalar property.
                type_.append((tokens[2].decode("ascii"), types[tokens[1]]))
            elif (len(tokens) == 5 and tokens[1] == b"list" and
                  tokens[2] in types and tokens[3] in types and _isvalidid(tokens[4])):
                # Store a list property.
                propname = tokens[4].decode("ascii")
                type_.append((propname, np.object_))
                list_types[name][propname] = (types[tokens[2]], types[tokens[3]])
            else:
                raise PLYError(f"Invalid PLY file (invalid element '{line}')")
            continue
        if name is not None:
            # Store the current element.
            elements[name] = np.empty((size,), dtype=type_)
        if line == b"end_header":
            break
        if tokens[0] == b"element":
            if len(tokens) != 3 or not _isvalidid(tokens[1]) or not tokens[2].isdigit():
                raise PLYError(f"Invalid PLY file (invalid element '{line}')")
            name = tokens[1].decode("ascii")
            type_ = []
            size = int(tokens[2])
            list_types[name] = {}
        else:
            line = line.decode("ascii")
            raise PLYError(f"Invalid PLY file (invalid header line '{line}')")
    return elements, list_types


def _read_ply_element_binary(f, element, list_types, big_endian):
    """Read an element from f using the binary format."""
    # 1) Build struct format strings corresponding to the data.
    # Strings for regular and list properties are interleaved.  This
    # is because the length of lists is known only after the
    # corresponding counter has been read.
    end = (">" if big_endian else "<")
    fmts = []
    fmt = []
    for j in range(len(element.dtype)):
        if np.issubdtype(element.dtype[j], np.object_):
            # List property: add the counter to the current format
            # string.  Then add a new string for the list items.  Then
            # start a new format string.
            lt, it = list_types[element.dtype.names[j]]
            fmt.append(_NP2STRUCT_MAP[lt])
            fmts.append(end + "".join(fmt))
            fmts.append(end + "{:d}" + _NP2STRUCT_MAP[it])
            fmt = []
        else:
            fmt.append(_NP2STRUCT_MAP[element.dtype[j].type])
    if fmt:
        fmts.append(end + "".join(fmt))
    # 2) Compute the size for each format string.  Use -1 to mark list items.
    sizes = [struct.calcsize(s) if i % 2 == 0 else -1 for i, s in enumerate(fmts)]
    # 3) Read the data.
    for i in range(element.shape[0]):
        row = []
        for fmt, sz in zip(fmts, sizes):
            if sz < 0:
                fmt = fmt.format(row[-1])
                sz = struct.calcsize(fmt)
                bytes_ = f.read(sz)
                row[-1] = list(struct.unpack(fmt, bytes_))
            else:
                row.extend(struct.unpack(fmt, f.read(sz)))
        element[i] = tuple(row)


def _read_numerical_ply_element_ascii(f, element):
    """Read an element without list properties from f using the ASCII format."""
    n = element.shape[0]
    k = 0
    while k < n:
        # Read in chunks to avoid doubling memory usage.
        lines = min(n - k, 128)
        arr = np.genfromtxt(f, dtype=element.dtype, max_rows=lines)
        if arr.shape[0] < lines:
            raise PLYError("PLY Error: unexpected EOF")
        element[k:k + lines] = arr
        k += lines


def _read_ply_element_ascii(f, element, list_types):
    """Read an element from f using the ASCII format."""
    if not element.dtype.hasobject:
        # Fast path for numerical-only elements.
        return _read_numerical_ply_element_ascii(f, element)
    for k in range(element.shape[0]):
        line = f.readline()
        if line is None:
            raise PLYError("PLY Error: unexpected EOF")
        tokens = line.split()
        data = []
        index = 0
        for j in range(len(element.dtype)):
            if index >= len(tokens):
                raise PLYError("PLY Error: invalid data line")
            if np.issubdtype(element.dtype[j], object):
                # Build the list.
                count = int(tokens[index])
                index += 1
                type_ = list_types[element.dtype.names[j]][1]
                lst = [type_(x) for x in tokens[index:index + count]]
                if len(lst) < count:
                    raise PLYError("PLY Error: invalid list")
                index += count
                data.append(lst)
            else:
                # Convert and append the scalar property.
                data.append(element.dtype[j].type(tokens[index]))
                index += 1
        element[k] = data


def _read_ply_handle(f):
    # Read the header.
    binary, big_endian = _read_header_intro(f)
    elements, list_types = _read_header_definitions(f)
    # Read the elements.
    for name, element in elements.items():
        if binary:
            _read_ply_element_binary(f, element, list_types[name], big_endian)
        else:
            _read_ply_element_ascii(f, element, list_types[name])
    return elements


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
    write_ply("a.ply", v, f, comments="test", binary=False)
    write_ply("b.ply", v, f, comments="test", binary=True)
    # print(buf.getvalue().decode("ascii"), end="")
    v1, f1, _ = read_ply("b.ply")


if __name__ == "__main__":
    _main()
