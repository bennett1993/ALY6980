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

st.set_page_config(page_title='BERT Model', layout="centered")
st.sidebar.header('BERT Model')

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

@st.cache_data()
def get_datasets(file):
    if file == None:
        new_resulting_factors = pd.read_csv("https://raw.githubusercontent.com/bennett1993/ALY6080/main/new_resulting_factors.csv")
        patient_info = pd.read_csv("https://raw.githubusercontent.com/bennett1993/ALY6080/main/patient_info.csv")
        word_banks = pd.read_csv("https://raw.githubusercontent.com/bennett1993/ALY6080/main/word_banks.csv")
    else:
        new_resulting_factors = pd.read_csv("https://raw.githubusercontent.com/bennett1993/ALY6080/main/new_resulting_factors.csv")
        patient_info = pd.read_csv("https://raw.githubusercontent.com/bennett1993/ALY6080/main/patient_info.csv")
        word_banks = pd.read_csv(file)
        
    return new_resulting_factors, patient_info, word_banks

with datasets: 
    
    file = st.file_uploader("If there are new word banks, please upload the current word_banks.csv file")
    
    new_resulting_factors, patient_info, word_banks = get_datasets(file)
    

with models:
    st.header('Model Training Time!')
    
    st.write('Please consult the following link for the KeyBERT extract_keywords function parameters: https://maartengr.github.io/KeyBERT/api/keybert.html#keybert._model.KeyBERT.extract_keywords')
    
    a = st.slider('Choose the upper bound for the number of words in each phrase', min_value=1, max_value=3, value=1, step=1)
    b = st.slider('Choose the number of keywords and phrases you would like', min_value=4, max_value=5, value=4, step=1)
    c = st.select_slider('Would you like to use Max Sum Distance for the determination of keywords and phrases?', options=['Yes', 'No Thank You'])
    
    if c == 'Yes':
        c = True
    elif c == 'No Thank You':
        c = False
        
    d = st.select_slider('Would you like to use Maximal Marginal Relevance (MMR) for the determination of keywords and phrases?', options=['Yes', 'No Thank You'])
    
    if d == 'Yes':
        d = True
    elif d == 'No Thank You':
        d = False
        
    e = st.slider('If you are using MMR, please choose the diversity of the keyword/keyphrase results:', min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    f = st.slider('If you are using Max Sum Distance, please choose the number of candidates to examine:', min_value=15, max_value=20, value=15, step=5)
    
    st.write('Here are your word banks for the chosen parameters:')
    
    word_banks[['patient_id', f"{a}_{b}_{c}_{d}_{e}_{f}"]]
    
    options = st.multiselect('Would you like to get word banks for specific patients?',word_banks['patient_id'])
    
    filtered_df = word_banks[word_banks['patient_id'].isin(options)]
    
    if options:
        filtered_df[['patient_id', f"{a}_{b}_{c}_{d}_{e}_{f}"]]
    else:
        word_banks[['patient_id', f"{a}_{b}_{c}_{d}_{e}_{f}"]]