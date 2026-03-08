import os
import sys
import dill

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as ex:
        raise CustomException(ex, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        # iterate through each model and optionally use corresponding params
        for model_name, model in models.items():
            # safely fetch parameters for this model (may be empty)
            model_params = params.get(model_name, {})
            # if you later want to perform grid search or set params, use model_params

            gs = GridSearchCV(model,model_params,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)


            #model.fit(X_train, y_train)  # Train Model

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as ex:
        raise CustomException(ex, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as ex:
        raise CustomException(ex, sys)