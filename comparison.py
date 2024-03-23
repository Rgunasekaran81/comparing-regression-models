import pandas as pd
from numpy import sqrt, max
from time import time

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score

import plotly.figure_factory as ff
import plotly.express as pe

from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

class comparison:

    def __init__(self, dataframe: pd, independent:list[str], dependent:str, usermodellist:list[str], dropattribute:list[str]=[], test_size=0.2, cvepoch:int=None, lassoalpha:float=0.1) -> None:
        self.dataframe = dataframe.drop(dropattribute, axis=1)
        self.independent = independent
        self.dependent  = dependent
        self.usermodellist = usermodellist
        self.cvepoch = cvepoch

        self.ridge_model = Ridge(True)
        self.linear_model = LinearRegression()
        self.lasso_model = Lasso(alpha = lassoalpha)
        self.xgboost_model = XGBRegressor()
        self.randomforest_reg = RandomForestRegressor()
        self.modelist = {"Ridge Regression":self.ridge_model, 
                        "Linear Regression":self.linear_model,
                        "Lasso Regression":self.lasso_model,
                        "Xgboost Regression":self.xgboost_model,
                        "Random Forest Regression":self.randomforest_reg
                        }

        #self.datafromat = self.cleandata()

        self.X = self.dataframe.drop(self.dependent, axis=1)
        self.y = self.dataframe[self.dependent]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=1)

    def eval_metrices(self, actual, pred) -> list[float]:
        rmse = sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        mse = mean_squared_error(actual, pred)
        score = r2_score(actual, pred)
        return [rmse, mae, mse, score]

    def crossvalidation(self, model_name, X_train, y_train) -> list[float]:
        cv_score = cross_val_score(estimator=model_name, X = X_train, y = y_train, cv=self.cvepoch)
        return cv_score.mean()

    def cleandata(self) -> dict[str:dict[str:int]]:
        def value_map(x, valuemap) -> int:
            for key, value in valuemap.items():
                if(x == key):
                    return value
        changein = {}
        for col in self.dataframe:
            values = self.dataframe[col].unique()
            if(type(values[0]) == str):
                valuemap = {}
                for num, val in enumerate(values):
                    valuemap[val] = num
                changein[col] = valuemap
                self.dataframe[col] = valuemap
                self.dataframe[col] = self.dataframe[col].apply(value_map, valuemap=valuemap)
        return changein

    def RegressionModels(self) -> dict[str:list[float]]:
        modeldata = {}
        for model in self.usermodellist:
            curmodel = self.modelist[model]
            start = time()
            curmodel.fit(self.X_train, self.y_train)
            end = time()-start
            y_pred = curmodel.predict(self.X_test)

            cval_score = self.crossvalidation(curmodel, self.X_train, self.y_train)
            modeldata[model] = [curmodel, self.eval_metrices(self.y_test, y_pred), cval_score, end]
        
        return modeldata

    def predict_y(self, data, usermodelist) -> dict[str:list[float]]:
        modelprediction = {}
        for model in usermodelist:
            prediction = self.modelist[model].predict(data)
            confidence = round(100 * (max(prediction[0])), 2)
            modelprediction[model] = [prediction, confidence]
        
        return modelprediction

    def plotgraph(self):
        figs = []
        figs.append(ff.create_distplot([self.dataframe[c] for c in self.dataframe.columns], self.dataframe.columns))
        figs.append(pe.box(self.dataframe))
        figs.append(pe.imshow(self.dataframe.corr(), text_auto=True))
        return figs

    def savemodel(self, model):
        filename = f"{model}.pkl"
        self.modelist[model].save(filename)

        return filename