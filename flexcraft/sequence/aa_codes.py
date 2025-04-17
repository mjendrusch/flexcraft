import numpy as np
import jax.numpy as jnp

AF2_AA_CODE = "ARNDCQEGHILKMFPSTWYV"
PMPNN_AA_CODE = sorted(AF2_AA_CODE)

def reindex_aatype(aatype, from_code, to_code):
    code_mapping = np.array([to_code.index(c) for c in from_code], dtype=np.int32)
    return code_mapping[aatype]

def decode_sequence(aatype: jnp.ndarray, code: str) -> str:
    return "".join([code[c] for c in aatype])
