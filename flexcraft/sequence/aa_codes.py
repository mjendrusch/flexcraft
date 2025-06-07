import numpy as np
import jax.numpy as jnp

AF2_CODE = "ARNDCQEGHILKMFPSTWYV"
PMPNN_CODE = sorted(AF2_CODE)

def translate(aatype, from_code, to_code):
    code_mapping = np.array([to_code.index(c) for c in from_code] + [20], dtype=np.int32)
    return code_mapping[aatype]

def decode(aatype: jnp.ndarray, code: str) -> str:
    code = code + "X"
    return "".join([code[c] for c in aatype])
