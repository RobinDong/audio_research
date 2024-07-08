""" Use grid search for all the audioaugmentation ops."""

import copy
import json
import subprocess


def execute(index, current):
    subprocess.run("python3 utils/generate_aug_esc50.py 1", shell=True)
    subprocess.run(f"python3 train.py 1 > {index}.log", shell=True)
    res = subprocess.run(
        f"grep -i avg {index}.log", shell=True, capture_output=True
    ).stdout
    res = str(res).split("\\n")[0]
    result = float(res.split(" ")[-1])
    print(current, result)


if __name__ == "__main__":
    with open("parameters.json", "r") as fp:
        obj = json.load(fp)

    config = {}
    for op_name in obj.keys():
        name = f"{op_name}.p"
        config[name] = [0.0]

    # first one
    with open("config.json", "w") as fp:
        json.dump(config, fp)
    execute(0, config)

    names = list(config.keys())
    index = 1
    for name in names:
        current = copy.deepcopy(config)
        for value in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            current[name] = [value]
            with open("config.json", "w") as fp:
                json.dump(current, fp)
            execute(index, current)
            index += 1
