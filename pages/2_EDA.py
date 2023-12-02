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

st.set_page_config(page_title='EDA', layout="centered")
st.sidebar.header('EDA')

datasets = st.container()
milestones = st.container()

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
    DevelopmentCategory = pd.read_csv("https://raw.githubusercontent.com/bennett1993/ALY6980/main/DevelopmentCategory.csv")
    Milestones = pd.read_csv("https://raw.githubusercontent.com/bennett1993/ALY6980/main/Milestones.csv")
    AgeGroup = pd.read_csv("https://raw.githubusercontent.com/bennett1993/ALY6980/main/AgeGroup.csv")
    Exercises = pd.read_csv("https://raw.githubusercontent.com/bennett1993/ALY6980/main/Exercises.csv")
        
    return DevelopmentCategory, Milestones, AgeGroup, Exercises
    
with datasets:
    DevelopmentCategory, Milestones, AgeGroup, Exercises = get_datasets()  

with milestones:
    st.header('Choose a Milestone')
    milestone_titles = Milestones['Title']
    option = st.selectbox('Choose a Milestone', milestone_titles)
    #exerciseID = Milestones.loc[option, 'ID']
    #age = Milestones.loc[option, 'AgeGroup']
    selected_row = Milestones[Milestones['Title'] == option]
    name_value = selected_row['ID'].iloc[0]
    st.write('exerciseID:', name_value)
    #st.write('age:', age)