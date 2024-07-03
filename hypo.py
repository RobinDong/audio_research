import sys
import json
import optuna
import subprocess

namespace = ""


def objective(trial):
    config = {
        "N": trial.suggest_int("Number of augs", 1, 6),
        "M": trial.suggest_int("Magnitude of augs", 1, 6),
    }

    with open("config.json", "w") as fp:
        json.dump(config, fp)
    subprocess.run(f"python3 utils/generate_aug_esc50.py {namespace}", shell=True)
    subprocess.run(f"python3 train.py {namespace} > {namespace}.log", shell=True)
    res = subprocess.run(
        f"grep -i avg {namespace}.log", shell=True, capture_output=True
    ).stdout
    print(f"[{res}]")
    res = str(res).split("\\n")[0]
    result = float(res.split(" ")[-1])
    return result


if __name__ == "__main__":
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
    study.optimize(objective, n_trials=20)
    print("Value: ", study.best_trial.value)
