from comparison import comparison
import streamlit as st
import pandas as pd 

csvfile = st.file_uploader(type="csv", label="upload csv file", accept_multiple_files=False)
if(csvfile != None):
    dataframe = pd.read_csv(csvfile)
    st.write(dataframe)
    availablemodel = ["Ridge Regression", 
                        "Linear Regression",
                        "Lasso Regression",
                        "Xgboost Regression",
                        "Random Forest Regression"]
    
    model = comparison(dataframe, dataframe.columns[-1])
    st.write(model.RegressionModels(availablemodel))

    

