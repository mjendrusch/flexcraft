'''Script for testing the AbsciBind protocol ipTM scoring.'''

from pathlib import Path
import json

import flexcraft.sequence.aa_codes as aas
from flexcraft.protocols.abscibind import AbsciBind
from flexcraft.files.pdb import PDBFile
from flexcraft.data.data import DesignData
import pandas as pd
import numpy as np
from flexcraft.structure.af import (
    AFInput, AFResult, get_model_haiku_params, model_config, make_af2, make_predict)
import jax
import jax.numpy as jnp
import gc
from urllib.request import urlretrieve
from datetime import datetime

def get_csv(ann:dict, directory:Path):
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

def get_pdb(pdb_id:str, directory:Path, base = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/pdb/", scheme="chothia"):
    if not directory.exists():
        directory.mkdir()
    file_path = directory/f"{pdb_id.lower()}_{scheme}.pdb"
    if file_path.exists():
        print(f"{file_path} exists ... skipping download.")
        return 0
    url = f"{base}{pdb_id.lower()}/?scheme={scheme}"
    print(f"Downloading from {url}...")
    urlretrieve(url, file_path)
    print(f"Download complete. File at {file_path}.")
    return 0

def load_data(out_dir:str|Path):
    '''Load testing data from annotation file.'''
    if not isinstance(out_dir, Path):
        out_dir=Path(out_dir)
    
    if not out_dir.exists():
        out_dir.mkdir()

    # check for annotation with info on files
    assert (out_dir/"annotation.json").exists(), "No annotation file!"
    with open(out_dir/"annotation.json", "r") as rf:
        annotations = json.load(rf)
    
    # check for files
    for n, ann in annotations.items():
        skip = False
        for key, value in ann.items():
            if not value and not key in ["folder",]:
                print(f"---{key} for {n} is missing! Skipping...---")
                skip = True
        if skip:
            continue
        ann_dir = out_dir/n
        if not ann_dir.exists():
            ann_dir.mkdir()
        if not (ann_dir/ann["file name"]).exists():
            print(f"Data File for {n} not in {ann_dir}, trying to fetch from annotation.")
            get_csv(ann, directory=ann_dir)
        else:
            print(f"Data File for {n} exists.")

        if not (ann_dir/f"{ann['PDB ID'].lower()}_chothia.pdb").exists():
            print(f"PDB file for {n} not found. Fetching from PDB.")
            get_pdb(ann["PDB ID"], directory=ann_dir)
        else:
            print(f"PDB file for {n} exists.")
    print("Checked all test files. Good to go...")

#load_data("flexcraft/data/o1_iptm_scoring")

def _insert_CDRs(
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
        lengths: lengths of CDRs in scaffoldzeros_like
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
    chain_end = indexes[chain_index == chain_id][-1]
    
    # sort by positions
    sorter = np.argsort(positions)
    cdrs = np.array(cdrs)[sorter] # type: ignore
    # add 0 to start the sequence for out
    lengths = np.concatenate([[0],np.array(lengths)[sorter]]) # type: ignore
    positions = np.concatenate([[0],np.array(positions)[sorter]]) # type: ignore
    positions = positions+chain_start

    #print("chain_start:\n",chain_start,"\npositions:\n",*positions, "\nlengths:\n",*lengths, "\npositions:\n",*positions)
    
    # convert inserts to designdata
    inserts = [DesignData.from_sequence(cdr).update(chain_index=jnp.full(len(cdr), chain_id)) for cdr in cdrs]

    out = DesignData.concatenate([
        scaffold[:chain_start],
        *[
        DesignData.concatenate([scaffold[positions[n]+lengths[n]:positions[n+1]], inserts[n]], sep_chains=False, sep_batch=False)
        for n in range(len(cdrs))
        ],
        scaffold[positions[-1]+lengths[-1]:],
        #scaffold[chain_end+1:],
        ],
        sep_chains=False, sep_batch=False)
    out = out.update(residue_index = jnp.arange(len(out["aa"])))

    mask = np.concatenate([
        mask[:chain_start],
        *[
        np.concatenate([mask[positions[n]+lengths[n]:positions[n+1]], np.zeros_like(inserts[n]["chain_index"], dtype=np.bool_)])
        for n in range(len(cdrs))
        ],
        mask[positions[-1]+lengths[-1]:]]
        )

    assert len(out["aa"]) == len(mask), "CDR mask does not have the same length as construct!"
    return out, mask

def insert_CDRs(
    cdrs:list[str]|tuple[str],
    chain_id:int|str,
    positions:list[tuple],
    scaffold:DesignData,):

    # check input
    assert len(cdrs) == len(positions)

    # sort by positions
    sorter = np.argsort([s for s,_ in positions])
    cdrs = np.array(cdrs)[sorter] # type: ignore
    # add 0 to start the sequence for out
    positions = np.array(positions)[sorter]

    inserts = [DesignData.from_sequence(cdr).update(chain_index=jnp.full(len(cdr), chain_id)) for cdr in cdrs]

    chain_mask = scaffold["chain_index"]==chain_id
    print(np.concat([np.arange(s,e) for s,e in positions]))
    index = np.arange(len(scaffold["aa"]))
    mask = (scaffold["residue_index"][:,None]==np.concat(
        [np.arange(s,e) for s,e in positions])
        ).any(axis=1)
    mask *= chain_mask
    l = []
    start = 0
    last = start
    for i, current in enumerate(mask):
        if last != current:
            if current:
                l.append(slice(start, i))
            else:
                start = i
        last = current
    l.append(slice(start,-1))
    print(l)
    out = DesignData.concatenate(
        [scaffold[l[0]],
        *[
            DesignData.concatenate([insert, scaffold[next_slice]], sep_batch=False, sep_chains=False)
        for insert, next_slice in zip(inserts, l[1:])
        ]],
        sep_batch=False,
        sep_chains=False,
    )
    ones = np.ones(len(scaffold["aa"]))
    target_mask = np.concatenate(
        [ones[l[0]],
        *[
            np.concatenate([np.zeros(len(insert)), ones[next_slice]])
        for insert, next_slice in zip(cdrs, l[1:])
        ]],
    )

    # update residue index
    out = out.update(residue_index = jnp.arange(len(out["aa"])))

    assert len(out["aa"]) == len(target_mask), "CDR mask does not have the same length as construct!"
    return out, target_mask


def clean_chothia(file):
    if isinstance(file, str):
        file = Path(file)
    out_path = Path(file.with_suffix("").__str__()+"_clean.pdb")
    with open(out_path, "w") as wf:
        with open(file, "r") as rf:
            l = "init value"
            while l:
                l = rf.readline()
                if l.startswith("ATOM"):
                    wf.write(l[:26]+" "+l[27:])
                elif not l.startswith("HETATM"):
                    wf.write(l)
    return out_path

def _clean_chothia(file:Path|str,
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
                        n = str(n)
                        n = (" "*(4-len(n))) + n
                        l = l[:22]+n+(" "*(11-len(n)))+l[33:]
                    
                elif current_chain != l[21]:
                    # add chain to chain order if chain changes
                    if l[21] in out.keys():
                        current_chain = l[21]
                        chain_order.append(current_chain)

                wf.write(l)
    return out, chain_order

def _convert_chains(chain_ids:np.ndarray):
    '''
    Convert str ids into numeric index.
    Notes:
        id numbers are derived from unique sorted ids
    '''
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])
    return chain_index


def abscibind_pipe(data_dir:str|Path,
    af_parameter_path,
    af2_key,
    targets:str|list|None = None,
    max_designs:None|int = None,
    num_recycle=0,
    clip_ab:bool=False,
    verbose:bool=False,
    cdr_positions:list=[ #start and stop indices of cdrs
        (26,26+9+1),
        (52,52+4+1),
        (95,95+7+1)
        ],
    **abscibind_kwargs):
    """Run abscibind on structures used to benchmark in origin1."""

    if verbose:
        print("Targets: ",targets)
    if isinstance(targets, str):
        targets = [targets]

    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)

    with open(data_dir/"annotation.json", "r") as rf:
        annotations = json.load(rf)

    abscibind = AbsciBind(key=af2_key, af_parameter_path=af_parameter_path, num_recycle=num_recycle, **abscibind_kwargs)

    def update_structure(data:DesignData, where:np.ndarray, save:Path|None):
        '''Predict structure at where.'''
        af_input = (AFInput
                        .from_data(data)
                        .add_template(data,where=where))
        if not "multimer" in abscibind.model[0]:
            num_chains = len(jnp.unique(data["chain_index"]))
            af_input = af_input.block_diagonal(num_sequences=num_chains)
                
        af_result:AFResult = abscibind.af2m(abscibind.af2_params[0], abscibind.key(), af_input)
        if save:
            af_result.save_pdb(save)
        return af_result.to_data()

    out_dir = data_dir/f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_af:{abscibind.model[0]}_clip:{clip_ab}_nrecycle:{num_recycle}"
    if not out_dir.exists():
        out_dir.mkdir(exist_ok=True)
    out_path = out_dir/"ipTM_data.csv"
    out_data = pd.DataFrame(index = [0],columns=["scaffold", "ipTM","default_iptm", "ab_iptm", "HCDR1","HCDR2","HCDR3","KD (nM)","Binder"])

    for scaffold_name, scaffold_ann in annotations.items():
        print(f"\n---{scaffold_name}---\n")
        if targets and not scaffold_name in targets:
            print(f"Skipping {scaffold_name}, as omitted from targets...")
            continue
        if not scaffold_ann["file name"]:
            print(f"Skip {scaffold_name} as file name missing.")
            continue
        
        scaffold_dir = data_dir/scaffold_name

        csv_path = scaffold_dir/scaffold_ann['file name']
        if not csv_path.exists():
            print(f"CSV file for {scaffold_name} is not in {scaffold_dir}. Expected name from annotations.json: {scaffold_ann['file name']}!")
            continue
        df = pd.read_csv(csv_path, header=0, index_col=None)
        if max_designs:
            df = df.iloc[:max_designs]
        ag_chain_id = scaffold_ann["Antigen Chain ID"]
        l_chain_id = scaffold_ann["Light Chain ID"]
        h_chain_id = scaffold_ann["Heavy Chain ID"]

        # load the protein structure
        pdb_path = scaffold_dir/f"{scaffold_ann['PDB ID'].lower()}_chothia.pdb"
        clean_path = clean_chothia(pdb_path)
        
        pdb = PDBFile(path=clean_path, convert_chains=False)
        data = pdb.to_data()
        if data["batch_index"].dtype != int:
            data = data.update(batch_index = np.zeros_like(data["chain_index"], dtype=np.int16))
        
        unique_chains = np.unique(data["chain_index"])
        data = data.update(chain_index = _convert_chains(data["chain_index"]))
        # convert id to int index, analogously to _convert_chains
        ag_chain_index = sorted(unique_chains).index(ag_chain_id)
        h_chain_index = sorted(unique_chains).index(h_chain_id)
        l_chain_index = sorted(unique_chains).index(l_chain_id)

        ab_selector = (data["chain_index"][:, None]==[h_chain_index, l_chain_index]).sum(axis=1)
        if "clip-target" in scaffold_ann.keys():
            selector = (data["residue_index"][:, None]==np.arange(*scaffold_ann["clip-target"])).sum(axis=1)
            selector *= ag_chain_index==data["chain_index"]
        else:
            selector = data["chain_index"] == ag_chain_index
        selector = (selector + ab_selector)>0
        data = data[selector]
        if clip_ab:
            # clip ab chains :120 residues
            index = jnp.arange(len(data["aa"]))
            h_start = index[data["chain_index"]==h_chain_index][0]
            h_stop = index[data["chain_index"]==h_chain_index][-1]
            if (h_stop-h_start)>120:
                data = data.index([slice(0,h_start+120,), slice(h_stop+1, -1)])
            index = jnp.arange(len(data["aa"]))
            l_start = index[data["chain_index"]==l_chain_index][0]
            l_stop = index[data["chain_index"]==l_chain_index][-1]
            if (l_stop-l_start)>120:
                data = data.index([slice(0,l_start+120,), slice(l_stop+1, -1)])
        if verbose:
            print("---scaffold data---",
                *[f"{k}:{v.shape}" for k,v in data.items()],
                f"ag_chain_index:{ag_chain_index}",
                f"ag_chain_id:{ag_chain_id}",
                f"sorted(unique_chains):{sorted(unique_chains)}",
                f"unique_chains:{unique_chains}",
                f"jnp.unique(data['chain_index']):{jnp.unique(data['chain_index'], return_counts=True)}",
                f"data['residue_index']:{data['residue_index']}",
                sep="\n")

        if len(data["aa"])>2000:
            print(f"{scaffold_name} has length >2000 aa! Skipping to avoid OOM...")
            continue

        # iterate over desgins and insert hcdrs
        for n, d_tuple in enumerate(df[["HCDR1", "HCDR2", "HCDR3","KD (nM)", "Binder"]].itertuples(index=False, name=None)):
            print(f"\n-Construct-",d_tuple, sep="\n")
            # insert CDRs
            data_conv, mask = insert_CDRs(cdrs=d_tuple[:3], chain_id=h_chain_index, positions=cdr_positions, scaffold=data)
            if verbose:
                print("---data_conv---",
                    *[f"{k}:{v.shape}" for k,v in data_conv.items()],
                    f"mask:{mask.shape}",
                    f"jnp.unique(data_conv['chain_index']):{jnp.unique(data_conv['chain_index'], return_counts=True)}",
                    f"data_conv['residue_index']:{data_conv['residue_index']}",
                    f"data_conv['chain_index']:{data_conv['chain_index']}",
                    sep="\n")

            # predict structure with insert
            save = None
            if verbose:
                save = out_dir/f"{scaffold_ann['PDB ID'].lower()}_pred_{n}.pdb"
            data_conv = update_structure(data_conv, where=mask, save=save)
            is_target = data_conv["chain_index"]==ag_chain_index
            if verbose:
                print("---updated structure---",
                    *[f"{k}:{v.shape}" for k,v in data_conv.items()],
                    f"is_target shape:{is_target.shape}",
                    f"is_target sum():{is_target.sum()}",
                    f"jnp.unique(data_conv['chain_index']):{jnp.unique(data_conv['chain_index'], return_counts=True)}",
                    f"data_conv['residue_index']:{data_conv['residue_index']}",
                    sep="\n")

            # calculate iptm
            iptm = abscibind(design=data_conv, is_target=is_target, save=save)
            if verbose:
                print(f"iptm:", *[f"{k}:{v}"for k,v in iptm[abscibind.model[0]].items()], sep="\n")

            out_data.loc[0,:] = [scaffold_name, *[v for v in iptm[abscibind.model[0]].values()], *d_tuple]
            out_data.to_csv(out_path,
            mode="a" if out_path.exists() else "w",
            header=~out_path.exists())
            
            gc.collect()

    return out_data

def predict_scaffold(
    ann_file:str|Path,
    af_parameter_path:str,
    af2_key,
    num_recycle:int=0,
    clip_ab:bool=False,
    verbose:bool=False,
    targets:None|list|str=None,
    **abscibind_kwargs,
    ):

    if not isinstance(ann_file, Path):
        ann_file = Path(ann_file)
    if isinstance(targets, str):
        targets = [targets]
    data_dir = ann_file.parent
    with open(ann_file, "r") as rf:
        annotations = json.load(rf)

    abscibind = AbsciBind(key=af2_key, af_parameter_path=af_parameter_path, num_recycle=num_recycle, **abscibind_kwargs)

    out_dir = data_dir/f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_fig13_af:{abscibind.model[0]}_clip:{clip_ab}_nrecycle:{num_recycle}"
    if not out_dir.exists():
        out_dir.mkdir()
    out_path = out_dir/"ipTM_data.csv"
    out_data = pd.DataFrame(index = [0],columns=["scaffold", "ipTM","default_iptm", "ab_iptm"])


    for scaffold_name, scaffold_ann in annotations.items():
        
        if targets and not scaffold_name in targets:
            print(f"Skipping {scaffold_name}, as omitted from targets...")
            continue
        if "skip" in scaffold_ann:
            print(f"Skipping {scaffold_name}, as {scaffold_ann['skip']}")
            continue
        scaffold_dir = data_dir/scaffold_name
        if not scaffold_dir.exists():
            scaffold_dir.mkdir()
            get_pdb(scaffold_name, scaffold_dir)

        ag_chain_id = scaffold_ann["Antigen Chain ID"]
        l_chain_id = scaffold_ann["Light Chain ID"]
        h_chain_id = scaffold_ann["Heavy Chain ID"]

        # load the protein structure
        pdb_path = scaffold_dir/f"{scaffold_name.lower()}_chothia.pdb"
        clean_path = clean_chothia(pdb_path)

        pdb = PDBFile(path=clean_path, convert_chains=False)
        data = pdb.to_data()
        if data["batch_index"].dtype != int:
            data = data.update(batch_index = np.zeros_like(data["chain_index"], dtype=np.int16))

        unique_chains = np.unique(data["chain_index"])

        data = data.update(chain_index = _convert_chains(data["chain_index"]))
        # convert id to int index, analogously to _convert_chains
        ag_chain_index = sorted(unique_chains).index(ag_chain_id)
        h_chain_index = sorted(unique_chains).index(h_chain_id)
        l_chain_index = sorted(unique_chains).index(l_chain_id)
    
        ab_selector = (data["chain_index"][:, None]==[h_chain_index, l_chain_index]).any(axis=1)
        if "clip-target" in scaffold_ann.keys():
            selector = (data["residue_index"][:, None]==np.arange(*scaffold_ann["clip-target"])).any(axis=1)
            selector *= ag_chain_index==data["chain_index"]
        else:
            selector = data["chain_index"] == ag_chain_index

        selector = (selector + ab_selector)>0
        data = data[selector]

        if clip_ab:
            # clip ab chains :120 residues
            index = jnp.arange(len(data["aa"]))
            h_start = index[data["chain_index"]==h_chain_index][0]
            h_stop = index[data["chain_index"]==h_chain_index][-1]
            if (h_stop-h_start)>120:
                data = data.index([slice(0,h_start+120,), slice(h_stop+1, -1)])
            index = jnp.arange(len(data["aa"]))
            l_start = index[data["chain_index"]==l_chain_index][0]
            l_stop = index[data["chain_index"]==l_chain_index][-1]
            if (l_stop-l_start)>120:
                data = data.index([slice(0,l_start+120,), slice(l_stop+1, -1)])

        if verbose:
            print("---scaffold data---",
                *[f"{k}:{v.shape}" for k,v in data.items()],
                f"ag_chain_index:{ag_chain_index}",
                f"ag_chain_id:{ag_chain_id}",
                f"sorted(unique_chains):{sorted(unique_chains)}",
                f"unique_chains:{unique_chains}",
                f"jnp.unique(data['chain_index']):{jnp.unique(data['chain_index'], return_counts=True)}",
                sep="\n")

        if len(data["aa"])>2000:
            print(f"{scaffold_name} has length >2000 aa! Skipping to avoid OOM...")
            continue
        
        is_target = data["chain_index"]==ag_chain_index

        data = data.update(residue_index = jnp.arange(len(data["aa"])))

        save=False
        if verbose:
            save = out_dir/f"{scaffold_name}_pred.pdb"

        iptm = abscibind(design=data, is_target=is_target, save=save)
        if verbose:
            print(f"iptm:", *[f"{k}:{v}"for k,v in iptm[abscibind.model[0]].items()], sep="\n")

        out_data.loc[0,:] = [scaffold_name, *[v for v in iptm[abscibind.model[0]].values()]]
        out_data.to_csv(out_path,
        mode="a" if out_path.exists() else "w",
        header=not out_path.exists())
        
        gc.collect()

    return out_data
