import pandas as pd
import  numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

class comparison:

    def __init__(self, dataframe: pd, dependent:str, dropattribute:list[str]=[], test_size=0.2, cvepoch:int=None, lassoalpha:float=0.1) -> None:
        self.dataframe = dataframe.drop(dropattribute, axis=1)
        self.dependent  = dependent 
        self.cvepoch = cvepoch

        self.ridge_model = Ridge(True)
        self.linear_model = LinearRegression()
        self.lasso_model = Lasso(alpha = lassoalpha)
        self.xgboost_model = XGBRegressor()
        self.randomforest_reg = RandomForestRegressor()

        X = self.dataframe.drop(self.dependent, axis=1)
        y = self.dataframe[self.dependent]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    def eval_metrices(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        mse = mean_squared_error(actual, pred)
        score = r2_score(actual, pred)
        return [rmse, mae, mse, score]

    def crossvalidation(self, model_name, X_train, y_train):
        cv_score = cross_val_score(estimator=model_name, X = X_train, y = y_train, cv=self.cvepoch)
        return cv_score.mean()

    def RidgeReg(self):
        self.ridge_model.fit(self.X_train, self.y_train)
        y_pred = self.ridge_model.predict(self.X_test)

        cval_score = self.crossvalidation(self.ridge_model, self.X_train, self.y_train)
        return [self.ridge_model, self.eval_metrices(self.y_test, y_pred), cval_score]

    def LinearReg(self):
        
        self.linear_model.fit(self.X_train, self.y_train)
        y_pred = self.linear_model.predict(self.X_test)

        cval_score = self.crossvalidation(self.linear_model, self.X_train, self.y_train)
        return [self.ridge_model, self.eval_metrices(self.y_test, y_pred), cval_score]

    def LassoReg(self):
        self.lasso_model.fit(self.X_train, self.y_train)
        y_pred = self.lasso_model.predict(self.X_test)
        
        cval_score = self.crossvalidation(self.lasso_model, self.X_train, self.y_train)
        return [self.ridge_model, self.eval_metrices(self.y_test, y_pred), cval_score]
    
    def XgboostReg(self):
        self.xgboost_model.fit(self.X_train, self.y_train)
        y_pred = self.xgboost_model.predict(self.X_test)
        
        cval_score = self.crossvalidation(self.xgboost_model, self.X_train, self.y_train)
        return [self.ridge_model, self.eval_metrices(self.y_test, y_pred), cval_score]

    def RandomForestReg(self):
        self.randomforest_reg.fit(self.X_train, self.y_train)
        y_pred = self.randomforest_reg.predict(self.X_test)

        cval_score = self.crossvalidation(self.randomforest_reg, self.X_train, self.y_train)
        return [self.randomforest_reg, self.eval_metrices(self.y_test, y_pred), cval_score]


df = pd.read_csv(r'D:\Projects\Comparing-Regression-Model\dataset\Admission_Predict.csv')
train = comparison(df, dependent='Chance of Admit ')
print(train.RandomForestReg())
print(train.LinearReg())
print(train.LassoReg())
print(train.RidgeReg())
print(train.XgboostReg())
