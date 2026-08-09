import os
import sys

from sklearn.neighbors import KNeighborsRegressor
from src.exception import CustomException
from src.logger import logging
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor  
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from dataclasses import dataclass
from src.utils import save_object
from src.logger import logging  
from src.exception import CustomException   
from xgboost import XGBRegressor
from src.utils import save_object, evaluate_model   
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "k-neighbors classifier": KNeighborsRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBclassifier": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report: dict = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
            )

            # To get the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name from the dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException(
                    "No best model found with a score greater than 0.6"
                )

            logging.info(f"Best found model on both training and testing dataset: {best_model}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            pridections = best_model.predict(X_test)
            r2_square = r2_score(y_test, pridections)
            return r2_square
        except Exception as e:
            raise CustomException(e, sys)