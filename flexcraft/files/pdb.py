import os
import shutil
import time
import uuid

from flexcraft.utils import load_pdb
from flexcraft.data.data import DesignData

class PDBFile:
    def __init__(self, data: DesignData | None=None, path=None, prefix="", tmpdir=None):
        self.prefix = prefix
        if tmpdir is None:
            tmpdir = "tmp/"
        self.tmpdir = tmpdir
        if path is None:
            path = self.get_tmp_path(prefix=prefix, tmpdir=tmpdir)
        self.path = path
        self.writable = False
        if data is None:
            data = load_pdb(path)
        else:
            self.writable = True
            data.save_pdb(path)
        self.data = data

    @staticmethod
    def get_tmp_path(prefix="", tmpdir="tmp/"):
        name = str(uuid.uuid4())
        if prefix:
            name = f"{prefix}_{name}"
        return f"{tmpdir}/{name}.pdb"

    def to_data(self):
        return self.data

    def clean(self):
        if not os.path.exists(self.tmpdir):
            os.makedirs(self.tmpdir)
        tmp_path = self.get_tmp_path(
            prefix=self.prefix, tmpdir=self.tmpdir)
        relevant_tags = ('ATOM', 'END', 'HETATM', 'LINK', 'MODEL', 'TER')
        with open(self.path, 'r') as f_in, open(tmp_path, 'wt') as f_out:
            for line in f_in:
                if line.startswith(relevant_tags):
                    f_out.write(line)
        shutil.move(tmp_path, self.path)
        return self

    def reload(self):
        if self.path is not None:
            self.data = load_pdb(self.path)
            return self.data
        raise ValueError("PDBFile is not attached to any path. Did you .remove it?")

    def remove(self):
        os.remove(self.path)
        self.path = None
        return self.data

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        raise NotImplementedError(
            "PDB-file data is immutable.")

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()
