import os
import sys
import dill
import optuna

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def run_optuna_study(objective_function, n_trials=5, seed=42):
    """
    Runs an Optuna study for a single model objective.
    Returns full study object.
    """

    try:
        sampler = optuna.samplers.TPESampler(seed=seed)

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )

        study.optimize(objective_function, n_trials=n_trials)

        return study

    except Exception as e:
        raise CustomException(e, sys)