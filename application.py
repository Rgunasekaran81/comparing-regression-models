from comparison import comparison
import streamlit as st
import pandas as pd 

csvfile = st.file_uploader(type="csv", label="upload csv file", accept_multiple_files=False)
if(csvfile != None):

    availablemodel = ["Ridge Regression", 
                        "Linear Regression",
                        "Lasso Regression",
                        "Xgboost Regression",
                        "Random Forest Regression"]
    dataframe = pd.read_csv(csvfile)

    section1 = st.columns(2)
    with section1[0]:
        st.write(dataframe)
    
    with section1[1]:
        checkboxes = st.columns(2)

        st.write("Select role of the attribute or drop the attribute")
        dependent = dataframe.columns[-1]
        independent = dataframe.columns[:-1]
        trackradio = {}

        for label in dataframe.columns:
            trackradio[label] = st.radio(label=label+":", options=["Independent", "Dependent", "Drop"], horizontal=True)
            if(label == dependent):
                trackradio[label] = st.radio(label=label+":", options=["Independent", "Dependent", "Drop"], index=1, horizontal=True)

    model = comparison(dataframe, dependent=dependent, independent=independent)
    st.write(model.RegressionModels(availablemodel))

    

