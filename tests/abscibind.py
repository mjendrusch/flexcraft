'''Script for testing the AbsciBind protocol ipTM scoring.'''

from pathlib import Path
import json

import flexcraft.sequence.aa_codes as aas
from flexcraft.protocols.AbsciBind import AbsciBind
from flexcraft.files.pdb import PDBFile
from flexcraft.data.data import DesignData
import pandas as pd
import numpy as np
from flexcraft.structure.af import (
    AFInput, AFResult, get_model_haiku_params, model_config, make_af2, make_predict)
import jax
import jax.numpy as jnp
from urllib.request import urlretrieve

def load_data(out_dir:str|Path):
    '''Load testing data from annotation file.'''
    if not isinstance(out_dir, Path):
        out_dir=Path(out_dir)
    
    if not out_dir.exists():
        out_dir.mkdir()

    def get_csv(ann:dict, directory:Path=out_dir):
        if not directory.exists():
            directory.mkdir()
        url = ann['github repository']+"/main"
        for n in ["folder", "file name"]:
            if ann[n]:
                url += f"/{ann[n]}"
        print(f"Downloading {url}...")
        urlretrieve(url, directory/ann["file name"])
        
        print(f"Download complete")
        return 1

    def get_pdb(pdb_id:str, directory:Path = out_dir, base = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/pdb/", scheme="chothia"):
        if not directory.exists():
            directory.mkdir()
        file_path = directory/f"{pdb_id.lower()}_{scheme}.pdb"
        if file_path.exists():
            print(f"{file_path} exists ... skipping download.")
            return 1
        url = f"{base}{pdb_id.lower()}/?scheme={scheme}"
        print(f"Downloading from {url}...")
        urlretrieve(url, file_path)
        print(f"Download complete. File at {file_path}.")
        return 1

    # check for annotation with info on files
    assert (out_dir/"annotation.json").exists(), "No annotation file!"
    with open(out_dir/"annotation.json", "r") as rf:
        annotations = json.load(rf)

    existing_files = [p for p in out_dir.glob("*.csv")]
    
    # check for files
    for n, ann in annotations.items():
        skip = False
        for key, value in ann.items():
            if not value and not key in ["folder",]:
                print(f"---{key} for {n} is missing! Skipping...---")
                skip = True
        if skip:
            continue
        if not ann["file name"] in existing_files:
            print(f"Data File for {n} not in {out_dir}, trying to fetch from annotation.")
            get_csv(ann)
        else:
            print(f"Data File for {n} exists.")

        if not (out_dir/f"{ann['PDB ID'].lower()}_chothia.pdb").exists():
            print(f"PDB file for {n} not found. Fetching from PDB.")
            get_pdb(ann["PDB ID"])
        else:
            print(f"PDB file for {n} exists.")
    print("Checked all test files. Good to go...")

#load_data("flexcraft/data/o1_iptm_scoring")

def insert_CDRs(
        cdrs:list[str]|tuple[str],
        chain_id:int|str,
        positions:list[int]|tuple[int],
        lengths:list[int]|tuple[int],
        scaffold:DesignData,
        ):
    """
    Function to insert HCDR designs into antibody scaffolds.
    Args:
        cdrs: list of str, contains strings for each CDR
        chain_ids: list of chain_index for each CDR
        positions: start position of each CDR within its chain
        lengths: lengths of CDRs in scaffold
        scaffold: DesignData object, scaffold for insertion
    Returns:
        Inserted data: DesignData object with manipulated Antibody structure
        insert mask: bool mask, False at inserts
    Note:
        Converts chain index to numerical values
    """
    # check input
    assert len(cdrs) == len(positions)
    assert len(cdrs) == len(lengths)


    chain_index = scaffold["chain_index"]
    mask = np.ones(len(chain_index), dtype=np.bool_)
    # get the chain start
    indexes = np.arange(len(chain_index))
    chain_start = indexes[chain_index==chain_id][0]
    positions = positions+chain_start
    
    # sort by positions
    sorter = np.argsort(positions)
    cdrs = np.array(cdrs)[sorter] # type: ignore
    # add 0 to start the sequence for out
    lengths = np.concatenate([[0],np.array(lengths)[sorter]]) # type: ignore
    positions = np.concatenate([[0],np.array(positions)[sorter]]) # type: ignore
    
    # convert inserts to designdata
    inserts = [DesignData.from_sequence(cdr) for cdr in cdrs]
    
    # convert scaffold to get numerical chain index
    conv_scaffold = scaffold.update(chain_index = _convert_chains(scaffold["chain_index"])) # type: ignore

    out = DesignData.concatenate([
        *[
        conv_scaffold[positions[n]+lengths[n]:positions[n+1]] + inserts[n]
        for n in range(len(cdrs))
        ],
        conv_scaffold[positions[-1]+lengths[-1]:]],
        sep_chains=False, sep_batch=False)
    
    mask = np.concatenate([
        *[
        np.concatenate([mask[positions[n]+lengths[n]:positions[n+1]], np.zeros_like(inserts[n]["chain_index"], dtype=np.bool_)])
        for n in range(len(cdrs))
        ],
        mask[positions[-1]+lengths[-1]:]]
        )

    assert len(out["aa"]) == len(mask)
    return out, mask

def clean_chothia(file:Path|str,
    cdr_pos:dict = {"HCHAIN":(np.array([26,52,95]),np.array([9,4,7])),
                    "LCHAIN":(np.array([24,50,89]),np.array([10,6,8]))},
    include:None|list = None):
    '''
    Clean a chothia structured pdb and return CDR positions.
    Args:
        file: pathlib.Path|str, relative or absolute path to the .pdb file
        cdr_pos: standard cdr positions and lengths
        include: list of chains to include in output
    Returns:
    - cdr_positions: dict, containing chains as key and tuple with positions (index in chain) and lengths of cdrs
    - chain order: list, containing chain keys in order of appearance
    '''
    if isinstance(file, str):
        file = Path(file)
    out_path = Path(file.with_suffix("").__str__()+"_clean.pdb")

    relevant_tags = ('ATOM', 'END', 'HETATM', 'LINK', 'MODEL', 'TER')
    out = {}
    gt = {}
    chain_order = []
    current_chain = "init value"
    stripped = "init value"
    with open(out_path, "w") as wf:
        with open(file, "r") as rf:
            l = "init value"
            while l:
                l = rf.readline()
                if not l.split(" ")[0] in relevant_tags:
                    wf.write(l)
                    if l.startswith("REMARK") and "PAIRED_HL" in l:
                        print(l)
                        # add chain pairing to output
                        s = l.split("PAIRED_HL")[-1].strip()
                        if include:
                            pairing = {c.split("=")[1]:c.split("=")[0] for c in s.split(" ")}
                            gt.update({k:cdr_pos.get(v, None) for k,v in pairing.items() if k in include})
                            # update out on deepcopy
                            out.update({k: (v[0].copy(), v[1].copy()) for k, v in 
                                ((k, cdr_pos.get(v, (np.array([]),np.array([])))) 
                                for k, v in pairing.items()) if k in include})
                        else:
                            pairing = {c.split("=")[1]:c.split("=")[0] for c in s.split(" ")}
                            gt.update({k:cdr_pos.get(v, None) for k,v in pairing.items()})
                            # update out on deepcopy
                            out.update({k: (v[0].copy(), v[1].copy()) for k, v in 
                                ((k, cdr_pos.get(v, (np.array([]),np.array([])))) 
                                for k, v in pairing.items())})
                    continue
                
                chain = l[21]
                if include:
                    if not chain in include:
                        continue
                if not l[22:31].strip().isnumeric():
                    # check if stripped same as last line
                    if stripped == l[22:31].strip():
                        stripped = l[22:31].strip()
                        n = stripped
                        while not n[-1].isnumeric():
                            n = n[:-1]
                        n = int(n)
                        n = str(n)
                        n = (" "*(4-len(n))) + n
                        l = l[:22]+n+(" "*(11-len(n)))+l[33:]
                        wf.write(l)
                        continue

                    stripped = l[22:31].strip()
                    # add stripped to cds
                    if stripped:
                        # get chathulu index
                        n = stripped
                        while not n[-1].isnumeric():
                            n = n[:-1]
                        n = int(n)
                        # elongate if in cds in out
                        # keep gt to check positions further downstream
                        out[chain][1][(gt[chain][0] <= n)&((gt[chain][0]+gt[chain][1])>=n)] += 1
                        # shift all further cds in out
                        # keep gt for downstream alignment
                        out[chain][0][gt[chain][0] > n] += 1
                        print(l, n)
                        n = str(n)
                        n = (" "*(4-len(n))) + n
                        l = l[:22]+n+(" "*(11-len(n)))+l[33:]
                        print(l)
                    
                elif current_chain != l[21]:
                    # add chain to chain order if chain changes
                    if l[21] in out.keys():
                        current_chain = l[21]
                        chain_order.append(current_chain)

                wf.write(l)
    return out, chain_order

def _convert_chains(chain_ids:np.ndarray):
    '''
    Convert str ids into numeric index if not int.
    Otherwise returns chain_ids.
    '''
    if chain_ids.dtype != int:
        unique_chain_ids = np.unique(chain_ids)
        chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
        chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])
        return chain_index
    return chain_ids


def abscibind_pipe(data_dir:str|Path,
    af_parameter_path,
    af2_key,
    targets:str|list|None = None,
    max_designs:None|int = None,
    **abscibind_kwargs):
    """Run abscibind on structures used to benchmark in origin1."""

    if isinstance(targets, str):
        targets = [targets]

    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)

    with open(data_dir/"annotation.json", "r") as rf:
        annotations = json.load(rf)

    abscibind = AbsciBind(key=af2_key, af_parameter_path=af_parameter_path, **abscibind_kwargs)

    def update_structure(data:DesignData, where:np.ndarray):
        '''Predict structure at where.'''
        af_input = (AFInput
                        .from_data(data)
                        .add_template(data,where=where))
        if not "multimer" in abscibind.model[0]:
            num_chains = len(jnp.unique(data["chain_index"]))
            af_input = af_input.block_diagonal(num_sequences=num_chains)
                
        af_result:AFResult = abscibind.af2m(abscibind.af2_params[0], abscibind.key(), af_input)
        return af_result.to_data()

    out_data = pd.DataFrame(columns=["scaffold", "ipTM", "HCDR1","HCDR2","HCDR3","KD (nM)","Binder"])

    for scaffold_name, scaffold_ann in annotations.items():
        if targets and not scaffold_name in targets:
            print(f"Skipping {scaffold_name}, as omitted from targets...")
            continue
        if not scaffold_ann["file name"]:
            print(f"Skip {scaffold_name} as file name missing.")
            continue
        csv_path = data_dir/scaffold_ann['file name']
        if not csv_path.exists():
            print(f"CSV file for {scaffold_name} is not in {data_dir}. Expected name from annotations.json: {scaffold_ann['file name']}!")
            continue
        df = pd.read_csv(csv_path, header=0, index_col=None)
        if max_designs:
            df = df.iloc[:max_designs]
        ag_chain_id = scaffold_ann["Antigen Chain ID"]
        ab_chain_ids = (scaffold_ann["Light Chain ID"], scaffold_ann["Heavy Chain ID"])
        
        # load the protein structure
        positions, _ = clean_chothia(data_dir/f"{scaffold_ann['PDB ID'].lower()}_chothia.pdb")
        pdb = PDBFile(path=data_dir/f"{scaffold_ann['PDB ID'].lower()}_chothia_clean.pdb", convert_chains=False)
        data = pdb.to_data()
        if data["batch_index"].dtype != int:
            data = data.update(batch_index = np.zeros_like(data["chain_index"], dtype=np.int16))
        positions = positions[ab_chain_ids[1]]
        # iterate over desgins and insert hcdrs
        for d_tuple in df[["HCDR1", "HCDR2", "HCDR3","KD (nM)", "Binder"]].itertuples(index=False, name=None):
            # insert CDRs
            data_conv, mask = insert_CDRs(cdrs=d_tuple[:3], chain_id=ab_chain_ids[1], positions=positions[0], lengths=positions[1], scaffold=data)
            # predict structure with insert
            
            data_conv = update_structure(data_conv, where=mask)
            # calculate iptm
            iptm = abscibind(design=data_conv, is_target=(data_conv["chain_index"]==ag_chain_id))
    
            out_data.iloc[len(out_data)] = [scaffold_name, iptm[abscibind.model[0]], *d_tuple]
    out_data.to_csv(data_dir/"ipTM_data.csv")
    return out_data
