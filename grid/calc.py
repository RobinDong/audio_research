""" Calculate average final accuracy for all the augmentation ops. """

import json

from collections import defaultdict

output = []
with open("log", "r") as fp:
    mapping = defaultdict(float)
    for line in fp:
        arr = line.split()
        target = arr[-1]
        string = " ".join(arr[:-1]).replace("'", '"')
        obj = json.loads(string)
        important_col = None
        for key, value in obj.items():
            if value[0] > 0.0:
                important_col = key.split(".")[0]
                break
        mapping[important_col] += float(target)

    for key, value in sorted(
        list(mapping.items()), key=lambda pair: pair[1], reverse=True
    ):
        print(key, value / 10.0)
