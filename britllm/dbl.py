# This will load a json list and return it as a list of dicts.

import json

def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            data.append(json.loads(line))

    return data