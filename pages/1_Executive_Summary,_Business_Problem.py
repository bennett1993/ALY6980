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

st.set_page_config(page_title='Choose a Milestone', layout="centered")
st.sidebar.header('Choose a Milestone')

datasets = st.container()
milestone = st.container()

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
def get_datasets():
    DevelopmentCategory = pd.read_excel("https://raw.githubusercontent.com/bennett1993/ALY6980/main/TablesForStreamlit.xlsx", sheet_name=DevelopmentCategory)
    Milestones = pd.read_excel("https://raw.githubusercontent.com/bennett1993/ALY6980/main/TablesForStreamlit.xlsx", sheet_name=Milestones)
    AgeGroup = pd.read_excel("https://raw.githubusercontent.com/bennett1993/ALY6980/main/TablesForStreamlit.xlsx", sheet_name=AgeGroup)
    Exercises = pd.read_excel("https://raw.githubusercontent.com/bennett1993/ALY6980/main/TablesForStreamlit.xlsx", sheet_name=Exercises)
        
    return DevelopmentCategory, Milestones, AgeGroup, Exercises
    
with datasets:
    DevelopmentCategory, Milestones, AgeGroup, Exercises = get_datasets()  

with milestone:
    st.header('Choose a Milestone')
    milestone_titles = Milestones['Title']
    option = st.selectbox('Choose a Milestone', milestone_titles)