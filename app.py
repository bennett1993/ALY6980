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
    
    patient_info = pd.read_csv("https://github.com/bennett1993/ALY6080/blob/main/Data/patient_info.csv")
    symptom_details = pd.read_csv("https://github.com/bennett1993/ALY6080/blob/main/Data/symptom_details.csv")
    
    file = st.file_uploader("Upload new data in one of the following tables: additional_notes.csv, new_resulting_factors.csv, or tbi_incident.csv (make sure file name matches exactly)")
    
    if file == None:
        additional_notes = pd.read_csv("https://github.com/bennett1993/ALY6080/blob/main/Data/additional_notes.csv")
        new_resulting_factors = pd.read_csv("https://github.com/bennett1993/ALY6080/blob/main/Data/new_resulting_factors.csv")
        tbi_incident = pd.read_csv("https://github.com/bennett1993/ALY6080/blob/main/Data/tbi_incident.csv")
    elif file.name == 'additional_notes.csv':
        additional_notes = pd.read_csv(file)
    elif file.name == 'new_resulting_factors.csv':
        new_resulting_factors = pd.read_csv(file)
    elif file.name == 'tbi_incident.csv':
        tbi_incident = pd.read_csv(file)
    else:
        st.write('Please upload a file with the name additional_notes.csv, new_resulting_factors.csv, or tbi_incident.csv')
        
        
with eda:
    fig = plt.figure(figsize=(16, 8))
    ax = sns.countplot(x="factor", data=new_resulting_factors, order=pd.value_counts(new_resulting_factors['factor']).iloc[:10].index)
    plt.xticks(rotation=90)
    plt.title('Top Ten Factor Frequencies')
    
    for p in ax.patches:
        height = p.get_height()
        ax.text(x = p.get_x()+(p.get_width()/2), 
        y = height+80, 
        s = '{:.0f}'.format(height), 
        ha = 'center')
 
    st.pyplot(fig)
    
    fig = plt.figure(figsize=(16, 8))
    ax = sns.countplot(x='subcategory', data = new_resulting_factors, order=new_resulting_factors['subcategory'].value_counts(ascending=False).index)
    plt.xticks(rotation=90)
    plt.title('Subcategory Frequencies')
    
    for p in ax.patches:
        height = p.get_height()
        ax.text(x = p.get_x()+(p.get_width()/2), 
        y = height+100, 
        s = '{:.0f}'.format(height), 
        ha = 'center') 
        
    st.pyplot(fig)
    
    df = new_resulting_factors.groupby('patient_about_id').size().reset_index(name='number_entries')
    df = df.sort_values(by='number_entries',ascending=False)
    d = dict([(y,x+1) for x,y in enumerate(sorted(set(df['patient_about_id'])))])
    df['patient_id']= df['patient_about_id'].map(d)
    df = df.iloc[:10]
    
    fig = plt.figure(figsize=(30, 8))
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
    st.header('Time to train the model!')
    st.text('Here you get to choose the parameters of the keyword extration model')
    
    
