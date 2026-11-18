"""Classes for reading and writing PDB files."""

import os
from pathlib import Path
import shutil
import tempfile

import jax

from flexcraft.data.data import DesignData
from flexcraft.utils.io import load_pdb


class PDBFile:
    """Manage input/output PDB files backed by canonical atom24 ``DesignData``."""

    def __init__(
        self,
        data: DesignData | dict | str | os.PathLike | None = None,
        path: str | os.PathLike | None = None,
        prefix: str = "",
        tmpdir: str | None = None,
        temporary: bool = False,
    ):
        self.prefix = prefix
        self.temporary = temporary
        self.tmpdir = tmpdir
        self.path = None
        self.writable = False
        self.data = None

        if isinstance(data, (str, os.PathLike)) and path is None:
            path = os.fspath(data)
            data = None

        if path is None:
            path = self.get_tmp_path(prefix=prefix, tmpdir=tmpdir)
        else:
            path = os.fspath(path)
        self.path = path

        if data is None:
            loaded = load_pdb(self.path)
            self.data = loaded
        else:
            if isinstance(data, dict):
                data = DesignData.from_dict(data)
            if "mask" in data.data and not any(
                isinstance(value, jax.core.Tracer) for value in data.data.values()
            ):
                data = data.unpad()
            self.writable = True
            data.save_pdb(self.path)
            self.data = data

    @staticmethod
    def get_tmp_path(prefix: str = "", tmpdir: str | None = None) -> str:
        """Create a unique temporary ``.pdb`` file in ``tmpdir`` (or system temp dir)."""
        if tmpdir is not None:
            os.makedirs(tmpdir, exist_ok=True)
        file_prefix = f"{prefix}_" if prefix else "flexcraft_"
        with tempfile.NamedTemporaryFile(
            mode="wt", prefix=file_prefix, suffix=".pdb", dir=tmpdir, delete=False
        ) as tmp_file:
            return tmp_file.name

    def to_data(self) -> DesignData:
        """Return the underlying ``DesignData`` object."""
        return self.data

    def clean(self) -> "PDBFile":
        """Strip non-coordinate records from the underlying PDB file."""
        tmp_path = self.get_tmp_path(prefix=self.prefix, tmpdir=self.tmpdir)
        relevant_tags = ("ATOM", "END", "HETATM", "LINK", "MODEL", "TER")
        with open(self.path, "rt") as f_in, open(tmp_path, "wt") as f_out:
            for line in f_in:
                if line.startswith(relevant_tags):
                    f_out.write(line)
        shutil.move(tmp_path, self.path)
        return self

    def reload(self) -> DesignData:
        """Reload the underlying PDB file from disk."""
        if self.path is not None:
            self.data = load_pdb(self.path)
            return self.data
        raise ValueError("PDBFile is not attached to any path. Did you .remove it?")

    def remove(self) -> DesignData:
        """Delete the underlying PDB file and return its ``DesignData``."""
        if self.path is not None:
            Path(self.path).unlink(missing_ok=True)
            self.path = None
        return self.data

    def __getitem__(self, index):
        """Index into the underlying ``DesignData`` object."""
        return self.data[index]

    def __setitem__(self, index, value):
        raise NotImplementedError("PDB-file data is immutable.")

    def __iter__(self):
        return iter(self.data)

    def __del__(self):
        if getattr(self, "temporary", False) and getattr(self, "path", None) is not None:
            Path(self.path).unlink(missing_ok=True)
            self.path = None

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()
