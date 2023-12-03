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
recommendations = st.container()

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

#@st.cache_data()
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

    option_row = Milestones[Milestones['Title'] == option]
    exerciseID = option_row['ID'].iloc[0]
    exerciseID_row = Exercises[Exercises['exerciseID'] == exerciseID]
    title_text = exerciseID_row['Title'].iloc[0]
    skills = Exercises[Exercises['exerciseID'] == exerciseID]['Skills']
    age = option_row['AgeGroup'].iloc[0]

    st.write('The exercise ID for the milestone you chose is: ', exerciseID)
    st.write('This title of this exercise is: ', title_text)
    st.write('The age group for the milestone you chose is: ', age, ' months old')
    st.write('The associated skills are: ', skills)

def contains_keywords(text, keywords):
    return any(keyword in text for keyword in keywords)

with recommendations:
    below = st.slider("How many months below your child's age do you want to receive recommended exercises for?",1,6,1)
    above = st.slider("How many months above your child's age do you want to receive recommended exercises for?",1,6,1)

    age_low = age - below
    age_high = age + above

    if age_low < 0:
        age_low = 0
    
    if age_high > 36:
        age_high = 36

    Exercises_filtered = Exercises[(Exercises['AgeGroup'] >= age_low) & (Exercises['AgeGroup'] <= age_high)]

    string_to_search = skills.str.cat(sep=' ')
    string_lower = string_to_search.lower()
    
    st.write('Determining recommended exercises. Takes a minute...')
    model = KeyBERT()
    keywords = [keyword for keyword, score in model.extract_keywords(string_lower,keyphrase_ngram_range=(1, 3))]

    matching_rows = Exercises_filtered[Exercises_filtered['Skills'].apply(lambda x: contains_keywords(x, keywords))]
    skills_combined = matching_rows.groupby('exerciseID').agg({'Skills': '; '.join, 'exerciseID':'first','Title': 'first', 'DevelopmentCategory': 'first', 'AgeGroup': 'first'})

    new_order = ['exerciseID','Title','DevelopmentCategory','AgeGroup','Skills']
    skills_combined = skills_combined.reindex(columns=new_order)

    st.write("Your child's recommended exercises are: ", skills_combined)



    
