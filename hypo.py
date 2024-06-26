import json
import optuna
import subprocess


def objective(trial):
    config = {
        "gaussian_min_amp": trial.suggest_float(
            "AddGaussianNoise: min_amp", 0.0001, 0.01, log=True
        ),
        "gaussian_max_amp": trial.suggest_float(
            "AddGaussianNoise: max_amp", 0.01, 0.1, log=True
        ),
        "ts_min_rate": trial.suggest_float("TimeStretch: min_rate", 0.5, 1.0),
        "ts_max_rate": trial.suggest_float("TimeStretch: max_rate", 1.0, 1.5),
        "ps_min_semi": trial.suggest_int("PitchShift: min_semi", -5, 0),
        "ps_max_semi": trial.suggest_int("PitchShift: max_semi", 0, 5),
        "shift_min": trial.suggest_float("Shift: min", -0.5, 0),
        "shift_max": trial.suggest_float("Shift: max", 0, 0.5),
    }

    with open("config.json", "w") as fp:
        json.dump(config, fp)

    subprocess.run(
        "python3 utils/generate_aug_esc50.py", shell=True, capture_output=True
    )
    subprocess.run("python3 train.py > log", shell=True, capture_output=True)
    print("finish")
    res = subprocess.run("grep -i avg log", shell=True, capture_output=True).stdout
    print(f"[{res}]")
    res = str(res).split("\\n")[0]
    print(f"[[{res}]]")
    result = float(res.split(" ")[-1])
    return result


if __name__ == "__main__":
    study_name = "audio_aug"
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{study_name}.db",
        direction="maximize",
    )
    study.optimize(objective, n_trials=20)
    print("Value: ", study.best_trial.value)
