import streamlit as st
import json
import os
import numpy as np
import pandas as pd
import altair as alt
import sys
from io import StringIO
from sklearn.preprocessing import minmax_scale

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(FILE_DIR)

sys.path.append(PARENT_DIR)
from utils import load_data_file, handle_na
from simrec import recommend

CONFIG_FILE = os.path.join(FILE_DIR, "config.json")
with open(CONFIG_FILE, "r") as fp:
    config = json.load(fp)

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode("utf-8")

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_submit():
    st.session_state.clicked = True

def load_data():
    if st.session_state.uploaded_file is not None:
        _, file_extension = os.path.splitext(st.session_state.uploaded_file.name)
        f = StringIO(st.session_state.uploaded_file.getvalue().decode("utf-8"))
        st.session_state.X = load_data_file(f, file_extension)

st.set_page_config(
    page_title="SIMREC App",
)

st.header("Welcome to the SIMREC App")
st.write("""SIMREC is a **SIM**ilarity measure **REC**ommendation system for mixed data clustering algorithms. 
This app allows you to use SIMREC recommend the best similarity pairs to use according to your own datasets.
Please refer to our [paper](https://dl.acm.org/doi/10.1145/3676288.3676302) if you want to know how the system works.""")

tab1, tab2 = st.tabs(["Use the App", "Get Help"])

with tab1:
    st.header("Inputs ​⚒️")
    # st.write("# Inputs ​⚒️​")

    uploaded_file = st.file_uploader("Input dataset: Choose a file in CSV or ARFF format", type=config["dataset_types"], key="uploaded_file", on_change=load_data)
    if "X" in st.session_state:
        st.dataframe(st.session_state.X, height=200)

        categorical_columns = st.multiselect(
            "Please indicate categorical columns.",
            st.session_state.X.columns,
            st.session_state.X.select_dtypes(exclude=["number"]).columns,
        )
        if len(categorical_columns) == 0:
            st.toast('The dataset should contain at least one categorical attribute', icon='🚨')
        if len(categorical_columns) == st.session_state.X.shape[1]:
            st.toast('The dataset should contain at least one numerical attribute', icon='🚨')


    col1, col2 = st.columns(2)
    with col1:
        algorithm = st.selectbox(
            "Choose an algorithm",
            config["algorithms"].keys(),
            format_func= lambda k: config["algorithms"][k]["name"]
        )
        submit = st.button("Submit", type="primary", key='submit', use_container_width=True, on_click=click_submit)

    with col2:
        cvi = st.selectbox(
            "Choose a cluster validity index",
            config["cvis"].keys(),
            format_func= lambda k: config["cvis"][k]["name"]
        )
        # st.write("You selected:", algorithm)

    st.header("Recommendation ​📊​")
    if submit:
        if "X" not in st.session_state:
            st.toast(f'Please select a dataset', icon='🚨')
            st.session_state.clicked = False
        else:
            st.session_state.X = handle_na(st.session_state.X)
            st.session_state.X = st.session_state.X.loc[:, (st.session_state.X != st.session_state.X.iloc[0]).any()]
            num_columns = [col for col in st.session_state.X.columns if col not in categorical_columns]
            cat_columns = [col for col in st.session_state.X.columns if col in categorical_columns]
            if len(num_columns) == 0 or len(cat_columns) == 0:
                st.toast(f'The dataset is suposed to be mixed. Got {len(num_columns)} numerical attribute(s) and {len(cat_columns)} categorical one (ones)', icon='🚨')
            else:
                Xnum = st.session_state.X.loc[:, num_columns]
                Xnum = Xnum.to_numpy()
                Xcat = st.session_state.X.loc[:, cat_columns]
                for col in Xcat.columns:
                    Xcat.loc[:, col] = pd.Categorical(Xcat.loc[:, col]).codes
                Xcat = Xcat.to_numpy(dtype=int)
                Xnum = minmax_scale(Xnum)
                recommendation = recommend(Xnum, Xcat, algorithm=algorithm, cvi=cvi)
                recommendation = pd.DataFrame(recommendation, columns=["similarity_pair", "score"])
                st.session_state.recommendation = recommendation

    if "recommendation" in st.session_state and st.session_state.clicked:
        col1, col2 = st.columns([1, 2])
        with col1:
            n_recommendations = st.number_input("Number of recommendations", min_value=config["min_recommendations"], max_value=config["max_recommendations"][algorithm], value=config["n_recommendations"])
        with col2:
            data = st.session_state.recommendation.iloc[:n_recommendations]
            # st.write(data)
            # data["similarity_pair"] = data["similarity_pair"].apply(lambda sp: "{}, {}".format(*sp.split('_')))
            # st.bar_chart(data.sort_values("score"), x="similarity_pair", y="score", x_label="Score", y_label="Similarity Pair", horizontal=True)
            chart = alt.Chart(data).mark_bar().encode(
                x=alt.X('score:Q', title="Score"),
                y=alt.Y('similarity_pair:O', sort=None, title="Similarity Pair"),
            )
            st.altair_chart(chart, use_container_width=True)
        with col1:
            csv = convert_df(data)
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="simrec_results.csv",
                mime="text/csv",
            )


with tab2:
    st.write("""
## When to use the SIMREC App ?
Use this app when you are about to run a given mixed data clustering algorithm on a mixed dataset and 
**want to know the best pair of numerical and categorical similarity measures to use in order 
to optimize the clustering performances.**

## How to use the App ?
Here are the diferrent steps to use the App.

### 1. Define your Inputs ​⚒️
The first step is to select the inputs of the recommendation system.
- **Dataset:** 
    - Upload the **mixed** dataset for which you want to perform recommendations. 
    - This should be a tabular dataset whith numerical and categorical attributes where lines 
    represent observations and columns represent attributes. 
    - Supported file format are **CSV** and **ARFF**.
- **Categorical attributes:** Once you have uploaded a dataset, an input widget will appear to specify 
the categorical attributes. The categorical attributes can be identified automatically 
if you provided the dataset in ARFF format.
- **Clustering Algorithm:** Select the clustering algorithm you want to use from the available ones.
- **Clusder Validity Index (CVI):** Select the CVI that you want to optimize.
### 2. Recommendation ​📊
Once you have defined your inputs, hit the submit button to compute the recommendations.
- The recommendations are presented as an ordered list of the similarity pairs according 
to ther predicted performances.
- Each similarity pair is presented in the following format: 
`<numeric similarity>_<categorical similarity>`, e.g., `euclidean_hamming` where the numeric 
similarity is the Euclidean distance and the categorical one is the Hamming distance
- You can define the number of similarity pair to recommend. For example if to select 10, 
only the top-10 similarity pairs in the ordered list will e displayed.
### 3. Download the recommendations ⬇️
Finally, you can decide to export or download the recommendations, i.e. the 
list of ordered similarity pairs with their predicted performances.
""")
