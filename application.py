from comparison import comparison
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import json

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
        with checkboxes[1]:
            st.write("Dependent Variable")
            
        checkboxs = {}
        for label in dataframe.columns:
            id = label.replace(" ","")
            if(id+"in" not in st.session_state):
                if(label != dataframe.columns[-1]):
                    st.session_state[id+"in"] = True
                else:
                    st.session_state[id+"in"] = False
            if(id+"de" not in st.session_state):
                if(label != dataframe.columns[-1]):
                    st.session_state[id+"de"] = False
                else:
                    st.session_state[id+"de"] = True

        for label in dataframe.columns:
            id = label.replace(" ","")
            with checkboxes[0]:
                if(st.session_state[id+"de"]):
                    checkboxs[id+"in"] = [st.checkbox(label=label, key=id+"in", disabled=True), label]
                else:
                    checkboxs[id+"in"] = [st.checkbox(label=label, key=id+"in"), label]
            with checkboxes[1]:
                if(st.session_state[id+"in"]):
                    checkboxs[id+"de"] = [st.checkbox(label=label, key=id+"de", disabled=True), label]
                else:
                    checkboxs[id+"de"] = [st.checkbox(label=label, key=id+"de"), label]

        #st.write(checkboxs)

        modeltrack = {}
        with checkboxes[0]:
            st.write("#")
            st.write("#")
            st.write("#")
            st.write("Select Regression models")
            for label in availablemodel:
                modeltrack[label] = st.checkbox(label=label, value=True)
            

        with checkboxes[1]:
            st.write("#")
            st.write("#")
            st.write("#")
            st.write("##")
            start = st.button("Start", type="primary", key="stren")
            st.write("#")
            stop = st.empty()
            stop.button("Stop", type="primary", disabled=True, key="stdis")
    if(start):
        
        with checkboxes[1]:
            stop = stop.button("Stop", type="primary", key="sten")

        modelist = []
        for key, val in modeltrack.items():
            if(val):
                modelist.append(key)

        dependent = ""
        independent = []
        drop = []
        for key, val in checkboxs.items():
            if(val[0]):
                if("de" in key):
                    dependent = val[1]
                if("in" in key):
                    independent.append(val[1])
            elif("in" in key):
                drop.append(val[1])
        
        drop.remove(dependent)
        
        st.write(dependent)
        model = comparison(dataframe, independent=independent, dependent=dependent, usermodellist=modelist, dropattribute=drop)

        tab1, tab2, tab3 = st.tabs(["Histogram", "Boxplot", "Heatmap"])
        with tab1:
            st.plotly_chart(model.plotgraph()[0], use_container_width=True)
        with tab2:
            st.plotly_chart(model.plotgraph()[1], use_container_width=True)
        with tab3:
            st.plotly_chart(model.plotgraph()[2], use_container_width=True)
        
        st.write(model.RegressionModels())