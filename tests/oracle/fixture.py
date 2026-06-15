"""Binary fixture format shared by the JAX oracle (writer) and the C++ tests (reader).

A fixture is a self-describing bag of named tensors. Layout (all little-endian):

    magic     : 4 bytes  b"FIXV"
    version   : u32       = 1
    n_tensors : u32
    repeated n_tensors times:
        name_len : u32
        name     : name_len bytes (utf-8, no null)
        dtype    : u32        0 = f32, 1 = i32
        rank     : u32
        dims     : rank * i64
        data     : prod(dims) elements, each 4 bytes (f32 or i32), row-major

Conventions used by the op-conformance suite:
    in0, in1, ...      forward inputs (the C++ side uploads these and builds the op)
    out                reference forward output
    grad_in0, ...      d(sum(out))/d(in_k) for each float input (gradient oracle)
    __tol_highest__    scalar f32: max abs error allowed at HIGHEST precision
    __tol_default__    scalar f32: max abs error allowed at DEFAULT (bf16) precision

This module is offline tooling only: it runs on the dev host with the JAX venv and
never enters the runtime. The C++ reader (cpp/fixture.hpp) is hermetic — it loads
the produced files and asserts, with no Python at test time.
"""
import struct
import numpy as np

MAGIC = b"FIXV"
VERSION = 1
DT_F32 = 0
DT_I32 = 1


def _dtype_code(arr):
    if arr.dtype == np.float32:
        return DT_F32
    if arr.dtype == np.int32:
        return DT_I32
    raise ValueError(f"unsupported fixture dtype {arr.dtype} (use float32 or int32)")


class FixtureWriter:
    """Accumulate named tensors, then .save(path)."""

    def __init__(self):
        self._tensors = []  # list of (name, np.ndarray)

    def add(self, name, arr):
        arr = np.ascontiguousarray(arr)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        if arr.dtype == np.int64:
            arr = arr.astype(np.int32)
        _dtype_code(arr)  # validate
        self._tensors.append((name, arr))
        return self

    def add_scalar(self, name, value):
        return self.add(name, np.asarray(value, dtype=np.float32))

    def save(self, path):
        with open(path, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<I", VERSION))
            f.write(struct.pack("<I", len(self._tensors)))
            for name, arr in self._tensors:
                nb = name.encode("utf-8")
                f.write(struct.pack("<I", len(nb)))
                f.write(nb)
                f.write(struct.pack("<I", _dtype_code(arr)))
                f.write(struct.pack("<I", arr.ndim))
                for d in arr.shape:
                    f.write(struct.pack("<q", int(d)))
                f.write(arr.tobytes(order="C"))


def read_fixture(path):
    """Reference reader (used by Python self-tests; the real consumer is C++)."""
    out = {}
    with open(path, "rb") as f:
        assert f.read(4) == MAGIC, "bad magic"
        (ver,) = struct.unpack("<I", f.read(4))
        assert ver == VERSION, f"version {ver}"
        (n,) = struct.unpack("<I", f.read(4))
        for _ in range(n):
            (name_len,) = struct.unpack("<I", f.read(4))
            name = f.read(name_len).decode("utf-8")
            (dt,) = struct.unpack("<I", f.read(4))
            (rank,) = struct.unpack("<I", f.read(4))
            dims = [struct.unpack("<q", f.read(8))[0] for _ in range(rank)]
            count = 1
            for d in dims:
                count *= d
            np_dt = np.float32 if dt == DT_F32 else np.int32
            data = np.frombuffer(f.read(count * 4), dtype=np_dt).reshape(dims)
            out[name] = data.copy()
    return out
