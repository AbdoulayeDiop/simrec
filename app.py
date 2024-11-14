import streamlit as st
import json
import os
import numpy as np
import pandas as pd
import altair as alt
from utils import load_data_file, handle_na
from io import StringIO
from simrec import recommend
from sklearn.preprocessing import minmax_scale
from dotenv import load_dotenv

# Load environment variables from the .env file (if present)
load_dotenv()

CONFIG_FILE = "app_config.json"
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
        filename, file_extension = os.path.splitext(st.session_state.uploaded_file.name)
        f = StringIO(st.session_state.uploaded_file.getvalue().decode("utf-8"))
        st.session_state.X = load_data_file(f, file_extension)

st.set_page_config(
    page_title="SIMREC App",
)

st.write("# Inputs ​⚒️​")
uploaded_file = st.file_uploader("Input dataset: Choose a file", type=config["dataset_types"], key="uploaded_file", on_change=load_data)
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

st.write("# Recommendation ​📊​")
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
            recommendation = recommend(Xnum, Xcat, os.getenv('models_dir'), algorithm=algorithm, cvi=cvi)
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


    # return {
    #     "id": data_id,
    #     "data_type": data_type,
    #     "numeric_attributes": num_columns,
    #     "categorical_attributes": cat_columns,
    #     "samples": X.index.values,
    #     "Xnum": Xnum,
    #     "Xcat": Xcat,
    #     "y": y,
    # }
