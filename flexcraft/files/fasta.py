import warnings

class FastaFile:
    def __init__(self, path=None, lines=None):
        self.path = path
        if lines is not None:
            self.lines = lines
            self.line_dict = {key: value for key, value in self.lines}
        else:
            self.lines = list()
            self.line_dict = dict()
        if self.path is not None:
            if lines is None:
                self.lines = _read_fasta(path)
                self.line_dict = {key: value for key, value in self.lines}
            else:
                for k, v in lines:
                    self.write_sequence(k, v)

    def append(self, name, sequence):
        self.lines.append((name, sequence))
        self.line_dict[name] = sequence

    def write(self, path):
        print("writing")
        self.path = path
        with open(self.path, "wt") as f:
            for k, v in self.lines:
                print(k, v)
                f.write(f">{k}\n{v}\n")

    def write_sequence(self, name, sequence):
        self.append(name, sequence)
        if self.path is not None:
            with open(self.path, "at") as f:
                f.write(f">{name}\n{sequence}\n")
        else:
            warnings.warn(
                "FastaFile has no path. "
                "Call self.write(path) to write to file.")

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.line_dict[index]
        return self.lines[index]
    
    def __len__(self):
        return len(self.lines)

    def items(self):
        return self.lines

def _read_fasta(path):
    with open(path, "rt") as f:
        header = None
        sequence = ""
        items = []
        for line in f:
            if line.startswith("#"):
                continue
            elif line.startswith(">"):
                if header is not None:
                    items.append((header, sequence))
                    header = None
                    sequence = ""
                header = line[1:].strip()
            else:
                sequence += line.strip()
        if header is not None:
            items.append((header, sequence))
        return items            
