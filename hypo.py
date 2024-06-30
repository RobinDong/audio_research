import sys
import json
import optuna
import subprocess

namespace = ""


def objective(trial):
    config = {
        "mp3_min_br": trial.suggest_int("Mp3Compression: min_br", 8, 16),
        "mp3_max_br": trial.suggest_int("Mp3Compression: max_br", 32, 64),
        "gaussian_min_amp": trial.suggest_float(
            "AddGaussianNoise: min_amp", 0.001, 0.01, log=True
        ),
        "gaussian_max_amp": trial.suggest_float(
            "AddGaussianNoise: max_amp", 0.01, 0.1, log=True
        ),
        "color_min_snr": trial.suggest_float(
            "AddColorNoise: min_snr", 1.0, 15.0, log=True
        ),
        "color_max_snr": trial.suggest_float(
            "AddColorNoise: max_snr", 15.0, 40.0, log=True
        ),
        "color_min_f": trial.suggest_float("AddColorNoise: min_f_decay", -11.0, -1.0),
        "color_max_f": trial.suggest_float("AddColorNoise: max_f_decay", 1.0, 11.0),
        "ts_min_rate": trial.suggest_float("TimeStretch: min_rate", 0.5, 0.9, step=0.1),
        "ts_max_rate": trial.suggest_float("TimeStretch: max_rate", 1.1, 1.5, step=0.1),
        "ps_min_semi": trial.suggest_int("PitchShift: min_semi", -3, 0),
        "ps_max_semi": trial.suggest_int("PitchShift: max_semi", 0, 3),
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
