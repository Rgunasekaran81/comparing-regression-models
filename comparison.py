import pandas as pd
from numpy import sqrt, max
from time import time
import pickle

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score

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

        self.dataframe = self.cleandata(self.dataframe)

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

    def cleandata(self, dataframe:pd) -> pd:
        for col in dataframe:
            values = dataframe[col].unique()
            if(type(values[0]) == str):
                valuemap = []
                for num in range(len(values)):
                    valuemap.append(num)
                dataframe[col].replace(values, valuemap, inplace=True)

        return dataframe

    def RegressionModels(self) -> dict[str:list[float]]:
        modeldata = {}
        for model in self.usermodellist:
            curmodel = self.modelist[model]
            start = time()
            curmodel.fit(self.X_train, self.y_train)
            endt = time()-start
            start = time()
            y_pred = curmodel.predict(self.X_test)
            endp = time()-start

            cval_score = self.crossvalidation(curmodel, self.X_train, self.y_train)
            modeldata[model] = [curmodel, self.eval_metrices(self.y_test, y_pred), cval_score, endt*100, endp*100]
        
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
        figs.append(pe.histogram(self.dataframe, marginal="box"))
        figs.append(pe.box(self.dataframe))
        figs.append(pe.imshow(self.dataframe.corr(), text_auto=True))
        return figs

    def savemodel(self, model):
        filename = f"{model}.pkl"
        with open(f'tempmodelsave\{filename}', "wb") as file:
            pickle.dump(self.modelist[model], file)
            
        return filename
    
    def comparisongraph(self, modelresult):
        strdata = []
        for key, val in modelresult.items():
            strdata.append({"MODEL":key,
                            "RMSE":val[1][0], 
                            "MAE":val[1][1], 
                            "MSE":val[1][2], 
                            "R2S":val[1][3], 
                            "CVS":val[2], 
                            "TIME TO TRAIN":val[3], 
                            "TIME TO PRED":val[4]
                            })
            
        strresult = pd.DataFrame(strdata)
        print(strresult)
        fig = pe.bar(strresult, x="MODEL", y=strresult.columns[1:])
        return fig