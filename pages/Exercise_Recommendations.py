import pandas as pd
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
import string
from keybert import KeyBERT
from PIL import Image

st.set_page_config(page_title='Exercise Recommendations', layout="centered")
st.sidebar.header('Exercise Recommendations')

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

def get_datasets():
    # reading in csv files on github
    DevelopmentCategory = pd.read_csv("https://raw.githubusercontent.com/bennett1993/ALY6980/main/DevelopmentCategory.csv")
    Milestones = pd.read_csv("https://raw.githubusercontent.com/bennett1993/ALY6980/main/Milestones.csv")
    AgeGroup = pd.read_csv("https://raw.githubusercontent.com/bennett1993/ALY6980/main/AgeGroup.csv")
    Exercises = pd.read_csv("https://raw.githubusercontent.com/bennett1993/ALY6980/main/Exercises.csv")
        
    return DevelopmentCategory, Milestones, AgeGroup, Exercises
    
with datasets:
    # calling function to get datasets
    DevelopmentCategory, Milestones, AgeGroup, Exercises = get_datasets()  

with milestones:
    st.header('Selecting Milestone and Determining Associated Characteristics')
    # storing title column
    milestone_titles = Milestones['Title']

    # creating dropdown input of milestone titles
    option = st.selectbox('Please choose a milestone', milestone_titles)

    # grabbing whole row from Milestones table
    option_row = Milestones[Milestones['Title'] == option]

    # grabbing fields from same row
    exerciseID = option_row['exerciseID'].iloc[0]
    Exercises_row = Exercises[Exercises['exerciseID'] == exerciseID]
    title_text = Exercises_row['Title'].iloc[0]
    skills = Exercises[Exercises['exerciseID'] == exerciseID]['Skills']
    age = option_row['AgeGroup'].iloc[0]

    # displaying attributes from same row
    st.write('The exercise ID for the milestone you chose is: ', exerciseID)
    st.write('This title of this exercise is: ', title_text)
    st.write('The age group for the milestone you chose is: ', age, ' months old')
    st.write('The associated skills are: ', skills)

# function that returns any matching keywords
def contains_keywords(text, keywords):
    return any(keyword in text for keyword in keywords)

with recommendations:
    st.header('Choosing Age Range for Recommended Exercises')
    # sliders to determine lower and upper bounds of age range
    below = st.slider("How many months below your child's age do you want to receive recommended exercises for?",1,6,1)
    above = st.slider("How many months above your child's age do you want to receive recommended exercises for?",1,6,1)

    # calculating lower and upper bounds
    age_low = age - below
    age_high = age + above

    # if the lower bound is below zero, set it to zero
    if age_low < 0:
        age_low = 0
    
    # if the upper bound is higher than 36, set it to 36
    if age_high > 36:
        age_high = 36

    # filter the Exercises dataframe to just include rows for which age is in previously defined range and not equal to current exercise ID
    Exercises_filtered = Exercises[(Exercises['AgeGroup'] >= age_low) & (Exercises['AgeGroup'] <= age_high) & (Exercises['exerciseID'] != exerciseID)]

    # concatenate all skills together and make lowercase
    string_to_search = skills.str.cat(sep=' ')
    string_lower = string_to_search.lower()

    st.write('The skills string is: ', string_lower)
    
    # sliders for extract_keywords parameters
    st.header('Choosing parameters of keyBERT model extract_keywords function')
    ngram_max = st.slider("Please choose the maximum number of words you want in keyword phrases",min_value=1,max_value=3,step=1,value=2)
    num_keywords = st.slider("Please choose the number of keywords and key phrases that you would like to return",min_value=5,max_value=20,step=1,value=10)

    # creating model object and extracting keywords from skills string
    st.header('Keywords and Key Phrases in Skills')
    st.write('Determining keywords and key phrases. Takes a minute...')
    model = KeyBERT()
    keywords = [keyword for keyword, score in model.extract_keywords(string_lower,keyphrase_ngram_range=(1, ngram_max),top_n=num_keywords)]
    st.write('The keywords and phrases in the associated skills are: ', keywords)

    st.header('Exercise Recommendations')
    st.write('Determining exercise recommendations. Takes a minute...')

    # using lambda function to call contains_keywords function
    matching_rows = Exercises_filtered[Exercises_filtered['Skills'].apply(lambda x: contains_keywords(x, keywords))]
    skills_combined = matching_rows.groupby('exerciseID').agg({'Skills': '; '.join, 'exerciseID':'first','Title': 'first', 'DevelopmentCategory': 'first', 'AgeGroup': 'first'})

    # re-ordering columns in dataframe
    new_order = ['exerciseID','Title','DevelopmentCategory','AgeGroup','Skills']
    skills_combined = skills_combined.reindex(columns=new_order)

    # printing recommended exercises
    st.write("Your child's recommended exercises are: ", skills_combined)