"""This module provides classes for handling common input and output CSV files in protein design scripts."""

class ScoreCSV:
    """Manages an output CSV file with protein design scores and metrics.

    Args:
        path: path to the output CSV file.
        keys: list of column names in the CSV file.
        default: optional string specifying the CSV entry for missing entries in a row.
    """
    def __init__(self, path, keys, default="none"):
        self.path = path
        self.keys = keys
        self.default = default
        with open(path, "wt") as f:
            f.write(",".join(self.keys) + "\n")

    def write_line(self, data: dict):
        """Write a row to the output CSV file given a data dictionary.
        Searches the `data` dictionary for keys in `self.keys` and writes the values
        to the corresponding columns in the CSV file, handling missing keys by writing
        `self.default`.

        Args:
            data: dictionary with one or more keys which are also in `self.keys`.
        """
        with open(self.path, "at") as f:
            result = []
            for key in self.keys:
                if key in data:
                    result.append(str(data[key]))
                else:
                    result.append(self.default)
            f.write(",".join(result) + "\n")
