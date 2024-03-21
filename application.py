from comparison import comparison
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import json
st.set_page_config(layout="wide")

print("test")
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
                id = label.replace(" ","")+"in"
                indeside += f'''<input type="checkbox" id="{id}"> <label for={id}>{label}</label> <br>'''
            st.markdown(indeside, unsafe_allow_html=True)
        
        with checkboxes[1]:
            st.write("Dependent Variable")
            for label in dataframe.columns:
                id = label.replace(" ","")+"de"
                deside += f'''<input type="checkbox" id="{id}"> <label for={id}>{label}</label> <br>'''
            st.markdown(deside, unsafe_allow_html=True)

        code = ""            
        for label in dataframe.columns:
            id1 = label.replace(" ","")
            if(label != dataframe.columns[-1]):
                code += '''
                    parent.document.getElementById('''+'"'+id1+'in'+'"'+''').checked = true;
                    parent.document.getElementById('''+'"'+id1+'de'+'"'+''').disabled = true;
                        '''
            else:
                code += '''
                    parent.document.getElementById('''+'"'+id1+'de'+'"'+''').checked = true;
                    parent.document.getElementById('''+'"'+id1+'in'+'"'+''').disabled = true;
                        '''
            code += '''
                checkboxin = parent.document.getElementById('''+'"'+id1+'in'+'"'+''');
                checkboxin.addEventListener('change', function() {
                    if (this.checked) {
                        parent.document.getElementById('''+'"'+id1+'de'+'"'+''').disabled = true;
                    } else {
                        parent.document.getElementById('''+'"'+id1+'de'+'"'+''').disabled = false;
                    }
                });
                checkboxde = parent.document.getElementById('''+'"'+id1+'de'+'"'+''');
                checkboxde.addEventListener('change', function() {
                    console.log(this.checked)
                    if (this.checked) {
                        parent.document.getElementById('''+'"'+id1+'in'+'"'+''').disabled = true;
                    } else {
                        parent.document.getElementById('''+'"'+id1+'in'+'"'+''').disabled = false;
                    }
                });
                    '''

        javascript = f'<script>parent.document.getElementsByTagName("iframe")[0].hidden="hidden";{code}</script>'
        components.html(javascript)

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

        code = """
            var dependent = null;
            var independent = [];
            var drop = []
            """
        for label in dataframe.columns:
            id = label.replace(" ", "")
            code += f"if(parent.document.getElementById('{id}in').checked)"
            code += """
                {
                    independent.push('"""+label+"""');
                }
                else
                {
                    drop.push('"""+label+"""');
                }
                """
            code += f"if(parent.document.getElementById('{id}de').checked)"
            code += """
                {
                    dependent = '"""+label+"""';
                }
                """
        
        code += "console.log(independent); console.log(dependent); console.log(drop);"
        javascript = f"<script>parent.document.getElementsByTagName('iframe')[1].hidden='hidden';{code}</script>"
        components.html(javascript)

        modelist = []
        for key, val in modeltrack.items():
            if(val):
                modelist.append(key)

        dependent = None
        independent = []
        drop = []
        with open("api.json", 'r') as api:
            data = json.load(api)
            dependent = data["dependent"]
            independent = data["independent"]
            drop = data["drop"]


        model = comparison(dataframe, dependent=dependent, independent=independent)
        st.write(model.RegressionModels(availablemodel))

    

