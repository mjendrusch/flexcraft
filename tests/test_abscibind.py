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

def load_data(out_dir:str|Path):
    from urllib.request import urlretrieve
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

    def get_pdb(pdb_id:str, directory:Path=out_dir, base="https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/pdb/", scheme="chothia"):
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
    assert (out_dir/"annotation.json").exists(), FileNotFoundError("No annotation file!")
    with open(out_dir/"annotation.json", "r") as rf:
        annotations = json.load(rf)

    existing_files = [p for p in out_dir.glob("*.csv")]
    
    # check for files
    for n, ann in annotations.items():
        con = False
        for key, value in ann.items():
            if not value and not key in ["folder",]:
                print(f"---{key} for {n} is missing! Skipping...---")
                con = True
        if con:
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

load_data("flexcraft/data/o1_iptm_scoring")

def insert_CDRs(cdrs:list|tuple, chain_ids:list|tuple, positions:list|tuple, lengths:list|tuple, scaffold:DesignData):
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

    """
    
    aa = scaffold["aa"]
    mask = np.ones(len(aa), dtype=np.bool_)
    chain_index = scaffold["chain_index"]
    indexes = np.arange(len(aa))
    for cdr, chain, position, length in zip(cdrs, chain_ids, positions, lengths):
        # compute chain start
        chain_start = indexes[chain_index==chain][0]
        # remove old and insert new
        aa = jnp.concat([aa[:chain_start+position],aas.encode(cdr, aas.AF2_CODE), aa[chain_start+position:length:]])
        mask = jnp.concat([mask[:chain_start+position],np.zeros(len(cdr), dtype=np.bool_), mask[chain_start+position:length:]])

    scaffold.update(aa=aa)
    return scaffold, mask

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
    relevant_tags = ('ATOM', 'END', 'HETATM', 'LINK', 'MODEL', 'TER')
    out = {}
    gt = {}
    chain_order = []
    current_chain = "init value"
    stripped = "init value"
    with open(file.with_suffix("").__str__()+"_clean.pdb", "w") as wf:
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

def abscibind_pipe(data_dir:str|Path, af_parameter_path, af2_key, **abscibind_kwargs):
    """Run abscibind on structures used to benchmark in origin1."""

    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)

    with open(data_dir/"annotations.json", "r") as rf:
        annotations = json.load(rf)

    abscibind = AbsciBind(key=af2_key, af_parameter_path=af_parameter_path, **abscibind_kwargs)

    def update_structure(data:DesignData, where:np.ndarray):
        '''Predict structure at where.'''
        af_input = (AFInput
                        .from_data(data)
                        .add_template(data,where=where))
        if not "multimer" in abscibind.af2_params[0]:
            num_chains = len(jnp.unique(data["chain_index"]))
            af_input = af_input.block_diagonal(num_sequences=num_chains)
                
        af_result:AFResult = abscibind.af2m(abscibind.af2_params[0], abscibind.key(), af_input)
        return af_result.to_data()

    out_data = pd.DataFrame(columns=["scaffold", "ipTM", "HCDR1","HCDR2","HCDR3","KD (nM)","Binder"])

    for scaffold_name, scaffold_ann in annotations.items():
        csv_path = data_dir/scaffold_ann['file name']
        if not csv_path.exists():
            print(f"CSV file for {scaffold_name} is not in {data_dir}. Expected name from annotations.json: {scaffold_ann['file name']}!")
            continue
        df = pd.read_csv(csv_path, header=0, index_col=None)
        ag_chain_id = scaffold_ann["Antigen Chain ID"]
        ab_chain_ids = (scaffold_ann["Light Chain ID"], scaffold_ann["Heavy Chain ID"])
        
        # load the protein structure
        positions, chain_starts = clean_chothia(data_dir/f"{annotations['PDB ID']}_chothia.pdb")
        pdb = PDBFile(path=data_dir/f"{annotations['PDB ID']}_chothia_clean.pdb")
        data = pdb.to_data()
        # filter chains
        chain_mask = data["chain"]==ag_chain_id|data["chain"]==ab_chain_ids[0]|data["chain"]==ab_chain_ids[1]
        data = data[chain_mask]
        positions = positions[ab_chain_ids[1]]
        cdr_chain_ids = (1,1,1)
        # make chain 0,1 or 2 for light, heavy and antigen
        chain_index = np.zeros_like(data["chain"])
        chain_index[data["chain"] == ab_chain_ids[1]] = 1
        chain_index[data["chain"] == ag_chain_id] = 2
        data = data.update(chain_index=chain_index)
        # iterate over desgins and insert hcdrs
        for d_tuple in df[["HCDR1", "HCDR2", "HCDR3","KD (nM)", "Binder"]].itertuples(index=False, name=None):
            # insert CDRs
            data, mask = insert_CDRs(cdrs=d_tuple[:3], chain_ids=cdr_chain_ids, positions=positions[0], lengths=positions[1], scaffold=data)
            # predict structure with insert
            data = update_structure(data, where=mask)
            # calculate iptm
            iptm = abscibind(design=data, is_target=(chain_index==2))
    
            out_data.iloc[-1] = [scaffold_name, [i for i in iptm.values()][0], *d_tuple]
    out_data.to_csv(data_dir/"ipTM_data.csv")
    return out_data

