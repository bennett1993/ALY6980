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

st.set_page_config(layout="centered")

header = st.container()
exec_summary = st.container()
business_prob = st.container()
dataset = st.container()
eda = st.container()
preprocessing = st.container()
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

with header:
    st.title('ALY 6080 XN Project')
    st.text('TBI Patient Keyword Extraction')

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


with dataset:
    st.header('Power of Patients Text Datasets')
    
    new_resulting_factors = pd.read_csv("https://raw.githubusercontent.com/bennett1993/ALY6080/main/new_resulting_factors.csv")
    patient_info = pd.read_csv("https://raw.githubusercontent.com/bennett1993/ALY6080/main/patient_info.csv")
    word_banks = pd.read_csv("https://raw.githubusercontent.com/bennett1993/ALY6080/main/word_banks.csv")
    
    file = st.file_uploader("If there are new word banks, please upload the current word_banks.csv")
    
    file      
        

with eda:
    st.header('Exploratory Data Analysis')
    
    first_chart = st.slider('Please choose the number of factors that you would like to see', min_value=1, max_value=20, value=10, step=1)
    
    fig = plt.figure(figsize=(16, 8))
    ax = sns.countplot(x="factor", data=new_resulting_factors, order=pd.value_counts(new_resulting_factors['factor']).iloc[:first_chart].index)
    plt.xticks(rotation=90)
    plt.title('Top f"{}" Factor Frequencies')
    
    for p in ax.patches:
        height = p.get_height()
        ax.text(x = p.get_x()+(p.get_width()/2), 
        y = height+80, 
        s = '{:.0f}'.format(height), 
        ha = 'center')
 
    st.pyplot(fig)
    
    second_chart = st.radio('Would you like the following chart in ascending or descending order?',('Ascending', 'Descending'))
    
    if second_chart == 'Ascending':
        second_chart = True
    elif second_chart == 'Descending':
        second_chart = False
    
    fig = plt.figure(figsize=(16, 8))
    ax = sns.countplot(x='subcategory', data = new_resulting_factors, order=new_resulting_factors['subcategory'].value_counts(ascending=second_chart).index)
    plt.xticks(rotation=90)
    plt.title('Subcategory Frequencies')
    
    for p in ax.patches:
        height = p.get_height()
        ax.text(x = p.get_x()+(p.get_width()/2), 
        y = height+100, 
        s = '{:.0f}'.format(height), 
        ha = 'center') 
        
    st.pyplot(fig)
    
    third_chart = st.slider('Please choose the number of patients that you would like to see', min_value=1, max_value=20, value=10, step=1)
    df = new_resulting_factors.groupby('patient_id').size().reset_index(name='number_entries')
    df = df.sort_values(by='number_entries',ascending=False)
    d = dict([(y,x+1) for x,y in enumerate(sorted(set(df['patient_id'])))])
    df['patient_id']= df['patient_id'].map(d)
    df = df.iloc[:third_chart]
    
    fig = plt.figure(figsize=(16, 8))
    ax = sns.barplot(x = 'patient_id', y = 'number_entries', data=df, order=df.sort_values('number_entries',ascending = False).patient_id)
    plt.title('Subcategory Frequencies')
    
    for p in ax.patches:
        height = p.get_height()
        ax.text(x = p.get_x()+(p.get_width()/2), 
        y = height+100, 
        s = '{:.0f}'.format(height), 
        ha = 'center') 
        
    st.pyplot(fig)
    
    fig = plt.figure(figsize=(10, 5))    
    df = patient_info.groupby(by='gender').size().reset_index(name='count')
    df = df.sort_values(by='count')
    plt.pie(df['count'],labels=df['gender'],autopct='%1.2f%%')
    plt.title("Gender Frequencies")
    
    st.pyplot(fig) 
    
    with models:
        
        slider = st.slider('Select', min_value=0.0, max_value=1.0, value=0.0, step=0.2)
        
        slider