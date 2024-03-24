from comparison import comparison
import streamlit as st
import pandas as pd

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
            start = [st.empty(), False]
            start[1] = start[0].button("Start", type="primary", key="stren")
            st.write("#")
            stop = st.empty()
            stop.button("Stop", type="primary", disabled=True, key="stdis")

    if(start[1]):
        with checkboxes[1]:
            start[0] = start[0].button("Start", type="primary", key="strdis", disabled=True)
            stop.button("Stop", type="primary", key="sten")

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

        modelist = []
        for key, val in modeltrack.items():
            if(val):
                modelist.append(key)

        model = comparison(dataframe, independent=independent, dependent=dependent, usermodellist=modelist, dropattribute=drop)

        tab1, tab2, tab3 = st.tabs(["Histogram", "Boxplot", "Heatmap"])
        with tab1:
            st.plotly_chart(model.plotgraph()[0], use_container_width=True)
        with tab2:
            st.plotly_chart(model.plotgraph()[1], use_container_width=True)
        with tab3:
            st.plotly_chart(model.plotgraph()[2], use_container_width=True)
        
        model_result = model.RegressionModels()

        Col1 = st.columns([14, 14, 14])
        Col2 = st.columns([14, 14, 14])

        def displayresult(i, j, Col):
            with Col[(i-j)]:
                btncols = st.columns([7, 7])
                
                with btncols[0]:
                    st.write(list(model_result.keys())[i])
                with btncols[1]:
                    filename = model.savemodel(list(model_result.keys())[i])
                    with open(f"tempmodelsave\{filename}", "rb") as file:
                        st.download_button(label="Download", data=file, file_name=filename)
                
                scores = list(model_result.values())[i]
                container = st.container(border=True)
                container.write(f":blue[**Root Mean Squared Error:**]     {scores[1][0]}")
                container.write(f":blue[**Mean Absolute Error:**]         {scores[1][1]}")
                container.write(f":blue[**Mean Squared Error:**]          {scores[1][2]}")
                container.write(f":blue[**R-squared:**]                   {scores[1][3]}")
                container.write(f":blue[**Cross validation:**]            {scores[2]}")
                container.write(f":blue[**Time take to train (x100):**]   {scores[3]}")
                container.write(f":blue[**Time take to predict (x100):**] {scores[4]}")

        for i in range(3):
            try:
                displayresult(i, 0, Col1)
            except:
                pass
        
        for i in range(3,5):
            try:
                displayresult(i, 3, Col2)
            except:
                pass

        st.plotly_chart(model.comparisongraph(model_result), use_container_width=True)
        
        restart = stop.button("Restart", type="primary", key="reen")
        if(restart):
            start[1] = True