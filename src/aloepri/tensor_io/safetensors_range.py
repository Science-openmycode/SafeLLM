from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

_DTYPE_BYTES: dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    path: Path
    dtype: str
    shape: tuple[int, ...]
    byte_offset: int
    byte_length: int


@dataclass(frozen=True)
class TensorOutputSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def byte_length(self) -> int:
        if self.dtype not in _DTYPE_BYTES:
            raise ValueError(f"unsupported Safetensors dtype: {self.dtype}")
        elements = 1
        for dimension in self.shape:
            if dimension < 0:
                raise ValueError(f"negative tensor dimension: {self.name}")
            elements *= dimension
        return elements * _DTYPE_BYTES[self.dtype]


class TensorSource(Protocol):
    def list_tensors(self) -> tuple[str, ...]: ...

    def metadata(self, name: str) -> TensorMetadata: ...

    def read_range(self, name: str, byte_offset: int, length: int) -> bytes: ...


class SafeTensorRangeSource:
    """Random-access Safetensors reader that never materializes a full tensor."""

    def __init__(self, model_dir_or_file: Path) -> None:
        self.root = model_dir_or_file
        self._tensors: dict[str, TensorMetadata] = {}
        for path in self._resolve_files(model_dir_or_file):
            self._index_file(path)
        if not self._tensors:
            raise ValueError(f"no tensors found: {model_dir_or_file}")

    @staticmethod
    def _resolve_files(path: Path) -> tuple[Path, ...]:
        if path.is_file():
            return (path,)
        index_path = path / "model.safetensors.index.json"
        if index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
            if not isinstance(weight_map, dict):
                raise ValueError("Safetensors index has no weight_map")
            return tuple(sorted({path / str(value) for value in weight_map.values()}))
        return tuple(sorted(path.glob("*.safetensors")))

    def _index_file(self, path: Path) -> None:
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise ValueError(f"invalid Safetensors file: {path}")
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length > 100 * 1024 * 1024:
                raise ValueError(f"Safetensors header exceeds 100 MiB: {path}")
            raw_header = handle.read(header_length)
        header = json.loads(raw_header)
        if not isinstance(header, dict):
            raise ValueError(f"invalid Safetensors header: {path}")
        data_start = 8 + header_length
        file_size = path.stat().st_size
        for name, raw_metadata in header.items():
            if name == "__metadata__":
                continue
            if name in self._tensors:
                raise ValueError(f"duplicate tensor in checkpoint: {name}")
            if not isinstance(raw_metadata, dict):
                raise ValueError(f"invalid tensor metadata: {name}")
            offsets = raw_metadata.get("data_offsets")
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise ValueError(f"invalid offsets for tensor: {name}")
            start, end = int(offsets[0]), int(offsets[1])
            if start < 0 or end < start or data_start + end > file_size:
                raise ValueError(f"tensor offsets exceed file: {name}")
            shape = tuple(int(value) for value in raw_metadata["shape"])
            dtype = str(raw_metadata["dtype"])
            expected = TensorOutputSpec(name, dtype, shape).byte_length
            if end - start != expected:
                raise ValueError(f"tensor byte length does not match shape: {name}")
            self._tensors[name] = TensorMetadata(
                name,
                path,
                dtype,
                shape,
                data_start + start,
                end - start,
            )

    def list_tensors(self) -> tuple[str, ...]:
        return tuple(sorted(self._tensors))

    def metadata(self, name: str) -> TensorMetadata:
        try:
            return self._tensors[name]
        except KeyError as error:
            raise KeyError(f"unknown tensor: {name}") from error

    def read_range(self, name: str, byte_offset: int, length: int) -> bytes:
        metadata = self.metadata(name)
        if byte_offset < 0 or length < 0 or byte_offset + length > metadata.byte_length:
            raise ValueError(f"range exceeds tensor boundary: {name}")
        with metadata.path.open("rb", buffering=0) as handle:
            handle.seek(metadata.byte_offset + byte_offset)
            payload = handle.read(length)
        if len(payload) != length:
            raise OSError(f"short tensor read: {name}")
        return payload


class SafeTensorRangeSink:
    """Preallocated, resumable Safetensors shard writer."""

    def __init__(
        self,
        final_path: Path,
        specs: tuple[TensorOutputSpec, ...],
        *,
        metadata: dict[str, str] | None = None,
    ) -> None:
        if not specs:
            raise ValueError("a Safetensors shard must contain at least one tensor")
        if len({spec.name for spec in specs}) != len(specs):
            raise ValueError("duplicate output tensor name")
        self.final_path = final_path
        self.partial_path = final_path.with_suffix(final_path.suffix + ".partial")
        self._specs = {spec.name: spec for spec in specs}
        self._offsets: dict[str, tuple[int, int]] = {}
        cursor = 0
        header: dict[str, object] = {}
        for spec in sorted(specs, key=lambda item: item.name):
            end = cursor + spec.byte_length
            self._offsets[spec.name] = (cursor, end)
            header[spec.name] = {
                "dtype": spec.dtype,
                "shape": list(spec.shape),
                "data_offsets": [cursor, end],
            }
            cursor = end
        if metadata:
            header["__metadata__"] = metadata
        raw_header = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
        padding = (-len(raw_header)) % 8
        self._raw_header = raw_header + (b" " * padding)
        self._data_start = 8 + len(self._raw_header)
        self._data_length = cursor
        self.partial_path.parent.mkdir(parents=True, exist_ok=True)
        if self.partial_path.exists():
            self._verify_existing_header()
        else:
            with self.partial_path.open("w+b") as handle:
                handle.write(struct.pack("<Q", len(self._raw_header)))
                handle.write(self._raw_header)
                handle.truncate(self._data_start + self._data_length)

    def _verify_existing_header(self) -> None:
        with self.partial_path.open("rb") as handle:
            existing_length = struct.unpack("<Q", handle.read(8))[0]
            existing = handle.read(existing_length)
        if existing_length != len(self._raw_header) or existing != self._raw_header:
            raise ValueError("partial shard header does not match current conversion plan")
        if self.partial_path.stat().st_size != self._data_start + self._data_length:
            raise ValueError("partial shard size does not match current conversion plan")

    def write_range(self, name: str, byte_offset: int, data: bytes) -> None:
        try:
            start, end = self._offsets[name]
        except KeyError as error:
            raise KeyError(f"unknown output tensor: {name}") from error
        if byte_offset < 0 or byte_offset + len(data) > end - start:
            raise ValueError(f"write exceeds tensor boundary: {name}")
        with self.partial_path.open("r+b", buffering=0) as handle:
            handle.seek(self._data_start + start + byte_offset)
            handle.write(data)

    def read_range(self, name: str, byte_offset: int, length: int) -> bytes:
        try:
            start, end = self._offsets[name]
        except KeyError as error:
            raise KeyError(f"unknown output tensor: {name}") from error
        if byte_offset < 0 or length < 0 or byte_offset + length > end - start:
            raise ValueError(f"read exceeds tensor boundary: {name}")
        with self.partial_path.open("rb", buffering=0) as handle:
            handle.seek(self._data_start + start + byte_offset)
            payload = handle.read(length)
        if len(payload) != length:
            raise OSError(f"short partial tensor read: {name}")
        return payload

    def flush(self) -> None:
        with self.partial_path.open("r+b", buffering=0) as handle:
            handle.flush()
            os.fsync(handle.fileno())

    def commit(self) -> str:
        self.flush()
        digest = _sha256_file(self.partial_path)
        os.replace(self.partial_path, self.final_path)
        return digest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
