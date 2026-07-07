import jax
import jax.numpy as jnp

from flexcraft.structure._colab_af.alphafold.common import residue_constants

def update_seq(seq, inputs, seq_1hot=None, seq_pssm=None, mlm=None):
  '''update the sequence features'''
  
  if seq_1hot is None: seq_1hot = seq 
  if seq_pssm is None: seq_pssm = seq
  target_feat = seq_1hot[0,:,:20]

  seq_1hot = jnp.pad(seq_1hot,[[0,0],[0,0],[0,22-seq_1hot.shape[-1]]])
  seq_pssm = jnp.pad(seq_pssm,[[0,0],[0,0],[0,22-seq_pssm.shape[-1]]])
  msa_feat = jnp.zeros_like(inputs["msa_feat"]).at[...,0:22].set(seq_1hot).at[...,25:47].set(seq_pssm)

  # masked language modeling (randomly mask positions)
  if mlm is not None:    
    X = jax.nn.one_hot(22,23)
    X = jnp.zeros(msa_feat.shape[-1]).at[...,:23].set(X).at[...,25:48].set(X)
    msa_feat = jnp.where(mlm[...,None],X,msa_feat)
    
  inputs.update({"msa_feat":msa_feat, "target_feat":target_feat})

def update_aatype(aatype, inputs):
  r = residue_constants
  a = {"atom14_atom_exists":r.restype_atom14_mask,
       "atom37_atom_exists":r.restype_atom37_mask,
       "residx_atom14_to_atom37":r.restype_atom14_to_atom37,
       "residx_atom37_to_atom14":r.restype_atom37_to_atom14}
  mask = inputs["seq_mask"][:,None]
  inputs.update(jax.tree_util.tree_map(lambda x:jnp.where(mask,jnp.asarray(x)[aatype],0),a))
  inputs["aatype"] = aatype

def expand_copies(x, copies, block_diag=True):
  '''
  given msa (N,L,20) expand to (1+N*copies,L*copies,22) if block_diag else (N,L*copies,22)
  '''
  if x.shape[-1] < 22:
    x = jnp.pad(x,[[0,0],[0,0],[0,22-x.shape[-1]]])
  x = jnp.tile(x,[1,copies,1])
  if copies > 1 and block_diag:
    L = x.shape[1]
    sub_L = L // copies
    y = x.reshape((-1,1,copies,sub_L,22))
    block_diag_mask = jnp.expand_dims(jnp.eye(copies),(0,3,4))
    seq = block_diag_mask * y
    gap_seq = (1-block_diag_mask) * jax.nn.one_hot(jnp.repeat(21,sub_L),22)  
    y = (seq + gap_seq).swapaxes(0,1).reshape(-1,L,22)
    return jnp.concatenate([x[:1],y],0)
  else:
    return x