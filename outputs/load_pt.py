"""Torch-free loader for torch.save() zip-format .pt files (fp32/fp16/bf16/int).

Reads tensor storages directly from the zip archive with a custom Unpickler.
Returns numpy arrays; dicts/lists/primitives pass through. Read-only use.
"""
import pickle
import zipfile
from collections import OrderedDict

import numpy as np

_DTYPES = {
    "FloatStorage": (np.float32, 4), "DoubleStorage": (np.float64, 8),
    "HalfStorage": (np.float16, 2), "BFloat16Storage": (np.uint16, 2),  # bf16 raw
    "LongStorage": (np.int64, 8), "IntStorage": (np.int32, 4),
    "ShortStorage": (np.int16, 2), "CharStorage": (np.int8, 1),
    "ByteStorage": (np.uint8, 1), "BoolStorage": (np.bool_, 1),
}


class _Storage:
    def __init__(self, data, np_dtype, is_bf16):
        self.data = data
        self.np_dtype = np_dtype
        self.is_bf16 = is_bf16


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
    arr = np.frombuffer(storage.data, dtype=storage.np_dtype)
    if storage.is_bf16:  # widen bf16 -> fp32
        arr = (arr.astype(np.uint32) << 16).view(np.float32)
    n = int(np.prod(size)) if size else 1
    arr = arr[storage_offset:storage_offset + n]
    itemstride = tuple(s * arr.itemsize for s in stride)
    return np.lib.stride_tricks.as_strided(arr, shape=tuple(size), strides=itemstride).copy() if size else arr[0]


class _FakeDtypeModule:  # torch.float32 etc. appearing as pickled globals
    def __getattr__(self, name):
        return name


class _Unpickler(pickle.Unpickler):
    def __init__(self, f, zf, prefix):
        super().__init__(f)
        self.zf, self.prefix = zf, prefix

    def find_class(self, module, name):
        if name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if name.endswith("Storage") and name in _DTYPES:
            return name  # marker used by persistent_load
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch" or module.startswith("torch."):
            return getattr(_FakeDtypeModule(), name)
        return super().find_class(module, name)

    def persistent_load(self, pid):
        typ, storage_name, key, location, numel = pid[0], pid[1], pid[2], pid[3], pid[4]
        assert typ == "storage"
        if not isinstance(storage_name, str):
            storage_name = storage_name.__name__ if hasattr(storage_name, "__name__") else str(storage_name)
        np_dtype, _ = _DTYPES[storage_name]
        raw = self.zf.read(f"{self.prefix}/data/{key}")
        return _Storage(raw, np_dtype, storage_name == "BFloat16Storage")


def load_pt(path):
    with zipfile.ZipFile(path) as zf:
        pkl = [n for n in zf.namelist() if n.endswith("/data.pkl")][0]
        prefix = pkl.rsplit("/", 1)[0]
        with zf.open(pkl) as f:
            return _Unpickler(f, zf, prefix).load()


if __name__ == "__main__":
    import sys
    obj = load_pt(sys.argv[1])
    def describe(o, d=0, key=""):
        pad = "  " * d
        if isinstance(o, dict):
            print(f"{pad}{key} dict[{len(o)}]")
            for k, v in list(o.items())[:40]:
                describe(v, d + 1, str(k))
        elif isinstance(o, np.ndarray):
            print(f"{pad}{key}: ndarray {o.shape} {o.dtype}")
        elif isinstance(o, (list, tuple)):
            print(f"{pad}{key}: {type(o).__name__}[{len(o)}] e.g. {str(o[:3])[:80]}")
        else:
            print(f"{pad}{key}: {type(o).__name__} = {str(o)[:100]}")
    describe(obj)

# alias expected by release scripts
# (load_pt defined above)
