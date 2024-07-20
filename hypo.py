import sys
import json
import optuna
import subprocess

namespace = ""


def objective(trial):
    with open("parameters.json", "r") as fp:
        obj = json.load(fp)

    config = {}
    for op_name, attrs in obj.items():
        for attr_name, attr in attrs.items():
            log = "log" in attr
            name = f"{op_name}.{attr_name}"
            if attr["dtype"] == "int":
                config[name] = (
                    trial.suggest_int(
                        name, attr["range"][0], attr["range"][1], log=log
                    ),
                )
            elif attr["dtype"] == "float":
                config[name] = (
                    trial.suggest_float(
                        name, attr["range"][0], attr["range"][1], log=log
                    ),
                )

    with open(f"config_{namespace}.json", "w") as fp:
        json.dump(config, fp)
    subprocess.run(f"python3 train.py {namespace} > {namespace}.log", shell=True)
    res = subprocess.run(
        f"grep -i avg {namespace}.log", shell=True, capture_output=True
    ).stdout
    print(f"## {res} ##")
    res = str(res).split("\\n")[0]
    result = float(res.split(" ")[-1])
    return result


if __name__ == "__main__":
    with open("parameters.json", "r") as fp:
        obj = json.load(fp)

    study_name = "audio_aug"
    if sys.argv[2] == "RND":
        sampler = optuna.samplers.RandomSampler()
    elif sys.argv[2] == "CMA":
        sampler = optuna.samplers.CmaEsSampler()
    elif sys.argv[2] == "GPS":
        sampler = optuna.samplers.GPSampler()
    else:
        sampler = optuna.samplers.TPESampler()

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{study_name}.db",
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
    )
    namespace = sys.argv[1]
    study.optimize(objective, n_trials=100)
    print("Value: ", study.best_trial.value)
