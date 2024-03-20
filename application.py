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
    checkboxes = st.columns(2)
    trackcheckbox = {} 
    with checkboxes[0]:
        st.write("select dependent attributes")
    for label in dataframe.columns:
        with checkboxes[0]:
            trackcheckbox[label+"0"] = st.checkbox(label=label, key=label+"0")
             
    with checkboxes[1]:
        st.write("select independent attributes")
    for label in dataframe.columns:
        with checkboxes[1]:
            trackcheckbox[label+"1"] = st.checkbox(label=label, key=label+"1")

    attributeselected = False
    dependent = []
    independent = None
   # model = comparison(dataframe, dataframe.columns[-1])
    #st.write(model.RegressionModels(availablemodel))

    

