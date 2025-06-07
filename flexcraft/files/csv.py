
class ScoreCSV:
    def __init__(self, path, keys, default="none"):
        self.path = path
        self.keys = keys
        self.default = default
        with open(path, "wt") as f:
            f.write(",".join(self.keys) + "\n")

    def write_line(self, data):
        with open(self.path, "at") as f:
            result = []
            for key in self.keys:
                if key in data:
                    result.append(str(data[key]))
                else:
                    result.append(self.default)
            f.write(",".join(result) + "\n")
