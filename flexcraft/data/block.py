from dataclasses import dataclass

@dataclass
class Partition:
    name: str
    id: int
    data: dict
    blocks: list
    def index(self, name):
        pass # TODO
    
    def __add__(self, other):
        pass

    def __truediv__(self, other):
        pass
    pass # TODO

def contig(cstring):
    cstring = "x=motif.pdb->10/#xA1-20/20/A1-20/"
    pass # TODO

def block(L=None):
    pass # TODO

def motif(pdb, contigs, group=0, name="motif"):
    # TODO
    result = [dict(
        atom_positions=...,
        atom_mask=...,
        aa=...,
        group=group,
        name=name
    ) for c in contigs]
    pass