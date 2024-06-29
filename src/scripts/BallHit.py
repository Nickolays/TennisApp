# Choose the best of (TrackNet, PoseEstimation, PathVanashingAlgorithm) 
# TrackNet is CNN which has a great quality
# PoseEstimation somnitelno
# PathVanashingAlgorithm - it's something new, We control ball position, and when it's disappears and the second frame appears ball poition.  
#  It not avialable now due to we feeling missing values


import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

import src

"""
        Idea 1
    Train a classification model predicting when was hitting the ball.

Predictors: pd.DataFrame
  - Shift_n, where n are shiffted values in diapason (-3, -2, -1, current, 1, 2, 3)
Output:
  - 0 or 1

"""

model_path = "models/hitting_model.cbm"
model = CatBoostClassifier().load_model(model_path)


class HittingModel:
    def __init__(self) -> None:
        pass

    def prepare(self, data:pd.DataFrame):
        pass

    def predict(self, data:pd.DataFrame, threshold=0.8):
        data = self.prepare(data)
        predict = self.model.predict_proba(data)[:, 1]
        if threshold:
            predict = np.where(predict > threshold, 1, 0)
        
        assert len(data) == len(predict)
        # Convert to list
        # assest type(data) == type(predict)
        return predict

    def evaluate(self, data:pd.DataFrame):
        pass
    

# IT CAN WORK 
# # PathVanashingAlgorithm - it's something new, We control ball position, and when it's disappears and the second frame appears ball poition.  
#  It not avialable now due to we feeling missing values
