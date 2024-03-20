from comparison import comparison
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
st.set_page_config(layout="wide")

csvfile = st.file_uploader(type="csv", label="upload csv file", accept_multiple_files=False)
if(csvfile != None):

    availablemodel = ["Ridge Regression", 
                        "Linear Regression",
                        "Lasso Regression",
                        "Xgboost Regression",
                        "Random Forest Regression"]
    dataframe = pd.read_csv(csvfile)

    section1 = st.columns([20, 20])
    with section1[0]:
        st.write(dataframe)
    
    with section1[1]:
        checkboxes = st.columns([10, 10])
        indeside = "<div id='indeside'>"
        deside = "<div id='deside'>"
        
        with checkboxes[0]:
            st.write("Independent Variable")
            for label in dataframe.columns:
                indeside += f'''<input type="checkbox" id="{label}in"> <label for={label}in>{label}</label> <br>'''
            st.markdown(indeside, unsafe_allow_html=True)
        
        with checkboxes[1]:
            st.write("Dependent Variable")
            for label in dataframe.columns:
                deside += f'''<input type="checkbox" id="{label}de"> <label for={label}de>{label}</label> <br>'''
            st.markdown(deside, unsafe_allow_html=True)

        javascript = '<script>parent.document.getElementsByTagName("iframe")[0].hidden="hidden";</script>'
        components.html(javascript)

    #model = comparison(dataframe, dependent=dependent, independent=independent)
    #st.write(model.RegressionModels(availablemodel))

    

