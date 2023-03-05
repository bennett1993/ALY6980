import pandas as pd
pd.set_option('display.max_columns', 500)
from pathlib import Path
import os
import glob
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)
import nbconvert
import numpy as np
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
from datetime import date as dt
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from keybert import KeyBERT
from PIL import Image

st.set_page_config(page_title='Home Page', layout="centered")
st.sidebar.header('Home Page')

header = st.container()
exec_summary = st.container()
business_prob = st.container()
datasets = st.container()
eda = st.container()
preprocessing = st.container()
models_considered = st.container()
models = st.container()
recommendations = st.container()
future_research = st.container()
other_relevant = st.container()
citations = st.container()

st.markdown(
    """
    <style>
    .main{
    background-color: #F5F5F5
    }
    </style>
    """,
    unsafe_allow_html=True
)

with preprocessing:
    st.header('Preprocessing Steps')
    
    image = Image.open('image8.png')
    st.image(image, caption='(Baheti, 2023)')
    
    st.markdown(
        """
        - Data Prep
            - Patient id's from all four dataframes were stacked on top of each other, then only unique patient id's were kept and stored in dataframe df1
            - Data used was the note column of additional_notes, description column of new_resulting_factors, factor column of registered_factors, and describe_event column of tbi_incident 
            - The four dataframes were grouped on patient_id, unique patient_id was determined, and then strings for each unique patient_id were combined with a space in between
            - New additional_notes was merged to df1 on patient_id to create master_data dataframe
            - Other three dataframes were merged with master_data on patient_id
            - Null and TRUE values in master_data were replaced with empty strings, all strings made lowercase, and stop words removed
        """)

with models_considered:
    st.header('Models Considered')
    image1 = Image.open('image9.jpeg')
    st.image(image1, caption='(Mall, 2021)')
    image2 = Image.open('image10.png')
    st.image(image2, caption='(Chenery-Howes, 2022)')
    image3 = Image.open('image11.jpeg')
    st.image(image3, caption='(Zhu et al., 2021)')
    
    st.markdown(
        """
        - LDA and LSI:
            - Pros:
                - Models are simple and easy to understand
                - Models are fast and can quickly mine text data
            - Cons:
                - Word banks were trivial and did not offer useful results
        - BERT:
            - Pros:
                - Model created meaningful word banks
                - BERT is a transformer model which is a neural network approach to NLP
                - Pretrained model using large data sets
            - Cons:
                - Black-Box model
                - Computationally expensive
        """)