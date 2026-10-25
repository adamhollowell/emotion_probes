"""Minimal torch-free loader for torch.save zip-format .pt files (tensors/lists/dicts only)."""
import zipfile, pickle, io
import numpy as np

_DTYPES = {
    "FloatStorage": np.float32, "DoubleStorage": np.float64,
    "HalfStorage": np.float16, "BFloat16Storage": "bf16",
    "LongStorage": np.int64, "IntStorage": np.int32,
    "ShortStorage": np.int16, "CharStorage": np.int8,
    "ByteStorage": np.uint8, "BoolStorage": np.bool_,
}

class _Storage:
    def __init__(self, dtype, key): self.dtype, self.key = dtype, key

class _StorageType:
    def __init__(self, name): self.name = name

def _rebuild_tensor_v2(storage, storage_offset, size, stride, *args):
    arr = storage["array"]
    if storage_offset: arr = arr[storage_offset:]
    itemsize = arr.dtype.itemsize
    byte_strides = tuple(s * itemsize for s in stride)
    return np.lib.stride_tricks.as_strided(arr, shape=tuple(size), strides=byte_strides).copy()

def load(path):
    zf = zipfile.ZipFile(path)
    names = zf.namelist()
    pkl_name = next(n for n in names if n.endswith("/data.pkl") or n == "data.pkl")
    prefix = pkl_name[: -len("data.pkl")]

    storages = {}
    def persistent_load(pid):
        typ, cls, key, location, numel = pid[0], pid[1], pid[2], pid[3], pid[4]
        assert typ == "storage"
        if key not in storages:
            raw = zf.read(f"{prefix}data/{key}")
            dt = _DTYPES[cls.name]
            if dt == "bf16":
                u = np.frombuffer(raw, dtype=np.uint16)
                arr = (u.astype(np.uint32) << 16).view(np.float32)
            else:
                arr = np.frombuffer(raw, dtype=dt)
            storages[key] = {"array": arr}
        return storages[key]

    class U(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
                return _rebuild_tensor_v2
            if module == "torch" and name in _DTYPES:
                return _StorageType(name)
            if module in ("collections", "builtins", "numpy", "numpy.core.multiarray"):
                return super().find_class(module, name)
            if (module, name) == ("torch", "device"):
                return lambda *a: None
            if (module, name) == ("torch.torch_version", "TorchVersion"):
                return str
            raise pickle.UnpicklingError(f"blocked: {module}.{name}")

    u = U(io.BytesIO(zf.read(pkl_name)))
    u.persistent_load = persistent_load
    return u.load()
