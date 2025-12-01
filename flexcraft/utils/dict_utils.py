import numpy as np

def pad_dict(data, num_residues):
    result = dict()
    for key, value in data.items():
        # larger lengths than num_residues are rejected
        if value.shape[0] > num_residues:
            raise ValueError(
                f"Array at key '{key}' of length {value.shape[0]} is longer than {num_residues}!")
        # if value is shorter than num_residues,
        # concatenate a zero-valued array of the difference
        if value.shape[0] < num_residues:
            difference = num_residues - value.shape[0]
            value = np.concatenate((
                value, np.zeros([difference] + list(value.shape[1:]), dtype=value.dtype)
            ), axis=0)
        result[key] = value
    return result
