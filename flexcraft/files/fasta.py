"""Utilities for reading and writing FASTA files."""

import os
import warnings


class FastaFile:
    """Container for reading and writing FASTA records."""

    def __init__(self, path=None, lines=None):
        self.path = path
        if lines is not None:
            self.lines = list(lines)
            self.line_dict = {key: value for key, value in self.lines}
        else:
            self.lines = []
            self.line_dict = {}
        if self.path is not None:
            if lines is None:
                self.lines = _read_fasta(path)
                self.line_dict = {key: value for key, value in self.lines}
            else:
                self.lines = []
                self.line_dict = {}
                for k, v in lines:
                    self.write_sequence(k, v)

    def append(self, name, sequence):
        self.lines.append((name, sequence))
        self.line_dict[name] = sequence

    def write(self, path):
        self.path = path
        directory = os.path.dirname(self.path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "wt") as handle:
            for k, v in self.lines:
                handle.write(f">{k}\n{v}\n")

    def write_sequence(self, name, sequence):
        self.append(name, sequence)
        if self.path is not None:
            directory = os.path.dirname(self.path)
            if directory and not os.path.isdir(directory):
                os.makedirs(directory, exist_ok=True)
            with open(self.path, "at") as handle:
                handle.write(f">{name}\n{sequence}\n")
        else:
            warnings.warn(
                "FastaFile has no path. Call self.write(path) to write to file."
            )

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.line_dict[index]
        return self.lines[index]

    def __len__(self):
        return len(self.lines)

    def items(self):
        return self.lines


def _read_fasta(path):
    with open(path, "rt") as handle:
        header = None
        sequence = ""
        items = []
        for line in handle:
            if line.startswith("#"):
                continue
            if line.startswith(">"):
                if header is not None:
                    items.append((header, sequence))
                    sequence = ""
                header = line[1:].strip()
            else:
                sequence += line.strip()
        if header is not None:
            items.append((header, sequence))
        return items
