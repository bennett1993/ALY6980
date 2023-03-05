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

st.set_page_config(page_title='Executive Summary, Business Problem', layout="centered")
st.sidebar.header('Executive Summary, Business Problem')

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

with exec_summary:
    st.header('Executive Summary')
    st.markdown(
        """
        - Sponsor is Power of Patients
            - Provide online portal for Traumatic Brain Injury (TBI) patients to track symptoms
        - Exploratory Data Analysis (EDA) of categorical variables in Python
        - Text mining algorithms considered:
            - Latent Dirichlet Allocation (LDA)
            - Latent Semantic Indexing (LSI)
            - Bidirectional Encoder Representations from Transformers (BERT)
        - Recommended model is BERT
        - Final deliverable: Streamlit application
            - Will serve as front-end for uploading new data and interacting with model
        """)

with business_prob:
    st.header('Business Problem')
    st.markdown(
        """
        - Symptom tracking system has many open-ended fields
            - Valuable information is hidden in these fields
        - Power of Patients needs to parse through open-ended fields and extract important information
        - Objective: develop word banks for each patient that hold key words and phrases
        - Word banks can be used to:
            - Evaluate how patients are responding to treatments and to prevent neuro-fatigue
            - Make exhaustive lists for dropdown fields, eliminating need for an open-ended 'other' field
            - Make suggestions for open-ended fields, so patients don’t have to type out everything (typing could be very difficult for them)
        """)