from comparison import comparison
import streamlit as st
import pandas as pd 

csvfile = st.file_uploader(type="csv", label="upload csv file", accept_multiple_files=False)
if(csvfile != None):
    dataframe = pd.read_csv(csvfile)
    st.write(dataframe)
    model = comparison(dataframe, "Sales")
    st.write(model.LassoReg())

    

