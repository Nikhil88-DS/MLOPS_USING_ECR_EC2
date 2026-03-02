import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, run_optuna_study
from src.components.data_transformation import DataTransformation




import os
import sys
from dataclasses import dataclass
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, run_optuna_study


@dataclass
class ModelTrainerConfig:
    best_model_path = os.path.join("artifacts", "best_model.pkl")
    leaderboard_path = os.path.join("artifacts", "model_leaderboard.csv")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            kf = KFold(n_splits=3, shuffle=True, random_state=42)

            results = []

            # -----------------------------------------
            # 🔥 Objective Functions Per Model
            # -----------------------------------------

            def rf_objective(trial):
                model = RandomForestRegressor(
                    n_estimators=trial.suggest_int("n_estimators", 10, 50),
                    max_depth=trial.suggest_int("max_depth", 3, 30),
                    min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                    random_state=42,
                    n_jobs=-1,
                )
                return cross_val_score(model, X_train, y_train, cv=kf, scoring="r2").mean()

            def xgb_objective(trial):
                model = XGBRegressor(
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    n_estimators=trial.suggest_int("n_estimators", 10, 50),
                    max_depth=trial.suggest_int("max_depth", 3, 10),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    random_state=42,
                    verbosity=0,
                    n_jobs=-1,
                )
                return cross_val_score(model, X_train, y_train, cv=kf, scoring="r2").mean()

            def cat_objective(trial):
                model = CatBoostRegressor(
                    depth=trial.suggest_int("depth", 4, 10),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    iterations=trial.suggest_int("iterations", 20, 80),
                    verbose=False,
                    random_state=42,
                )
                return cross_val_score(model, X_train, y_train, cv=kf, scoring="r2").mean()

            def gb_objective(trial):
                model = GradientBoostingRegressor(
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    n_estimators=trial.suggest_int("n_estimators", 10, 50),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    random_state=42,
                )
                return cross_val_score(model, X_train, y_train, cv=kf, scoring="r2").mean()

            def ada_objective(trial):
                model = AdaBoostRegressor(
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
                    n_estimators=trial.suggest_int("n_estimators", 10, 50),
                    random_state=42,
                )
                return cross_val_score(model, X_train, y_train, cv=kf, scoring="r2").mean()

            def dt_objective(trial):
                model = DecisionTreeRegressor(
                    max_depth=trial.suggest_int("max_depth", 3, 30),
                    min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                    random_state=42,
                )
                return cross_val_score(model, X_train, y_train, cv=kf, scoring="r2").mean()

            # Linear Regression (no tuning)
            linear_model = LinearRegression()
            linear_score = cross_val_score(
                linear_model, X_train, y_train, cv=kf, scoring="r2"
            ).mean()

            results.append({
                "model": "LinearRegression",
                "best_cv_score": linear_score,
                "best_params": {}
            })

            # -----------------------------------------
            # 🚀 Run Optuna Per Model
            # -----------------------------------------

            model_objectives = {
                "RandomForest": rf_objective,
                "XGB": xgb_objective,
                "CatBoost": cat_objective,
                "GradientBoosting": gb_objective,
                "AdaBoost": ada_objective,
                "DecisionTree": dt_objective,
            }

            for model_name, objective in model_objectives.items():
                logging.info(f"Tuning {model_name}...")

                study = run_optuna_study(objective, n_trials=5, seed=42)

                results.append({
                    "model": model_name,
                    "best_cv_score": study.best_value,
                    "best_params": study.best_params,
                })

            # -----------------------------------------
            # 📊 Leaderboard
            # -----------------------------------------

            leaderboard = pd.DataFrame(results)
            leaderboard = leaderboard.sort_values(
                by="best_cv_score", ascending=False
            )

            os.makedirs("artifacts", exist_ok=True)
            leaderboard.to_csv(self.config.leaderboard_path, index=False)

            logging.info("Model Leaderboard:")
            logging.info(leaderboard)

            # -----------------------------------------
            # 🏆 Select Best Model
            # -----------------------------------------

            best_model_row = leaderboard.iloc[0]
            best_model_name = best_model_row["model"]
            best_params = best_model_row["best_params"]

            model_map = {
                "RandomForest": RandomForestRegressor,
                "XGB": XGBRegressor,
                "CatBoost": CatBoostRegressor,
                "GradientBoosting": GradientBoostingRegressor,
                "AdaBoost": AdaBoostRegressor,
                "DecisionTree": DecisionTreeRegressor,
                "LinearRegression": LinearRegression,
            }

            if best_model_name == "LinearRegression":
                final_model = LinearRegression()
            else:
                final_model = model_map[best_model_name](
                    **best_params,
                    random_state=42
                )

            # Train best model
            final_model.fit(X_train, y_train)

            predictions = final_model.predict(X_test)
            test_r2 = r2_score(y_test, predictions)

            logging.info(f"Final Test R2 Score: {test_r2}")

            save_object(self.config.best_model_path, final_model)

            return test_r2

        except Exception as e:
            raise CustomException(e, sys)
