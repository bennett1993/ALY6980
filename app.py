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
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install(keybert)

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
    
    patient_info = pd.read_csv("https://raw.githubusercontent.com/bennett1993/ALY6080/main/Data/patient_info.csv")
    symptom_details = pd.read_csv("https://raw.githubusercontent.com/bennett1993/ALY6080/main/Data/symptom_details.csv")
    
    file = st.file_uploader("Upload new data in one of the following tables: additional_notes.csv, new_resulting_factors.csv, or tbi_incident.csv (make sure file name matches exactly)")
    
    if file == None:
        additional_notes = pd.read_csv("https://raw.githubusercontent.com/bennett1993/ALY6080/main/Data/additional_notes.csv")
        new_resulting_factors = pd.read_csv("https://raw.githubusercontent.com/bennett1993/ALY6080/main/Data/new_resulting_factors.csv")
        tbi_incident = pd.read_csv("https://raw.githubusercontent.com/bennett1993/ALY6080/main/Data/tbi_incident.csv")
    elif file.name == 'additional_notes.csv':
        additional_notes = pd.read_csv(file)
        new_resulting_factors = pd.read_csv("https://raw.githubusercontent.com/bennett1993/ALY6080/main/Data/new_resulting_factors.csv")
        tbi_incident = pd.read_csv("https://raw.githubusercontent.com/bennett1993/ALY6080/main/Data/tbi_incident.csv")
    elif file.name == 'new_resulting_factors.csv':
        additional_notes = pd.read_csv("https://raw.githubusercontent.com/bennett1993/ALY6080/main/Data/additional_notes.csv")
        new_resulting_factors = pd.read_csv(file)
        tbi_incident = pd.read_csv("https://raw.githubusercontent.com/bennett1993/ALY6080/main/Data/tbi_incident.csv")
    elif file.name == 'tbi_incident.csv':
        additional_notes = pd.read_csv("https://raw.githubusercontent.com/bennett1993/ALY6080/main/Data/additional_notes.csv")
        new_resulting_factors = pd.read_csv("https://raw.githubusercontent.com/bennett1993/ALY6080/main/Data/new_resulting_factors.csv")
        tbi_incident = pd.read_csv(file)
    else:
        st.write('Please upload a csv file with the name additional_notes.csv, new_resulting_factors.csv, or tbi_incident.csv')        
        
with eda:
    st.header('Exploratory Data Analysis')
    
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
    
    
with preprocessing:
    today = dt.today()
    patient_attributes = pd.merge(left=patient_info, right = symptom_details, left_on="patient_id", right_on="patient_id") 

    patient_attributes['date_of_birth_clean'] = pd.to_datetime(np.where(pd.to_datetime(patient_attributes['date_of_birth']).dt.year > today.year, 
                                                        (pd.to_datetime(patient_attributes['date_of_birth']).dt.year - 100).astype(str) + '-' + pd.to_datetime(patient_attributes['date_of_birth']).dt.month.astype(str) + '-' + pd.to_datetime(patient_attributes['date_of_birth']).dt.day.astype(str), 
                                                        patient_attributes['date_of_birth']))

    patient_attributes['age'] =  (pd.to_datetime(today) - pd.to_datetime(patient_attributes['date_of_birth_clean']))
    patient_attributes['age'] = round((patient_attributes['age'].astype(str).str.split(' ',expand=True)[0].astype(int))/365.25,2)
    
    patient_attributes = patient_attributes[['patient_id', 'age', 'gender', 'patient_type', 'id', 'details']]

    patient_attributes_notes = pd.merge(left=additional_notes, right = patient_attributes, left_on="patient_id", right_on="patient_id") 
    patient_attributes_notes = patient_attributes_notes[['id_x', 'patient_id', 'note', 'logged_at', 'age', 'gender', 'patient_type', 'details','additional_notes_date']]

    master_data = pd.merge(left=new_resulting_factors, right = patient_attributes_notes, left_on=["symptom_date", "patient_about_id"], right_on=["additional_notes_date", "patient_id"]) 
    master_data = master_data[['patient_id', 'patient_type', 'age', 'gender',  'details', 
                            'had_symptom', 'severity', 'description', 'factor', 'category', 'subcategory',
                            'logged_at_x',  'note']]
    # Fill Nas
    master_data['had_symptom'] = master_data['had_symptom'].fillna('False')
    master_data['description'] = master_data['description'].fillna('False')
    master_data['factor'] = master_data['factor'].fillna('None')
    master_data['category'] = master_data['category'].fillna('None')
    master_data['subcategory'] = master_data['subcategory'].fillna('None')
    master_data['note'] = master_data['note'].fillna('None')
    master_data['severity'] = master_data['severity'].fillna(master_data['severity'].mean())
    master_data = master_data.rename(columns={'logged_at_x':'log_time'})  
    master_data['agg_patient'] = master_data['patient_type'] + master_data['factor'] + master_data['subcategory']

    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    
    def clean(doc):
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized

    master_data.note_clean = master_data.note.apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))


    doc_clean = [clean(doc).split() for doc in master_data.note]  

    master_data.note = doc_clean
    
with models:
    st.header('BERT Model')

    word_bank = list()
    row = list()

    kw_model = KeyBERT()

    t = -1

    for i in range(len(master_data['note'])):
        
        word_bank.append(kw_model.extract_keywords(master_data['note'][i]))
        
        t = t + 1
        
        row.append(t)
    
    
    word_bank2 = word_bank.copy()
    key_word_dict = {'index':row,'key_words':word_bank2,'key_word_1':None,'key_word_2':None,'key_word_3':None,'key_word_4':None,'key_word_5':None}
    key_word_dataframe = pd.DataFrame(key_word_dict)

    for i in range(len(key_word_dataframe)):
        try: 
            key_word_dataframe['key_word_1'][i] = key_word_dataframe['key_words'][i][0][0][0]
            key_word_dataframe['key_word_2'][i] = key_word_dataframe['key_words'][i][1][0][0]
            key_word_dataframe['key_word_3'][i] = key_word_dataframe['key_words'][i][2][0][0]
            key_word_dataframe['key_word_4'][i] = key_word_dataframe['key_words'][i][3][0][0]
            key_word_dataframe['key_word_5'][i] = key_word_dataframe['key_words'][i][4][0][0]
        except:
            pass
    
    patient_words = pd.merge(left=master_data, right=key_word_dataframe, left_index=True, right_index=True, how='left')
    patient_words = patient_words[['patient_id', 'key_word_1', 'key_word_2', 'key_word_3', 'key_word_4', 'key_word_5']].drop_duplicates()

    patient_words = patient_words.dropna(subset=['key_word_1', 'key_word_2'])


    patient_words['key_word_3'] = patient_words['key_word_3'].fillna(value=pd.np.nan).fillna(" ")
    patient_words['key_word_4'] = patient_words['key_word_4'].fillna(value=pd.np.nan).fillna(" ")
    patient_words['key_word_5'] = patient_words['key_word_5'].fillna(value=pd.np.nan).fillna(" ")



    patients = pd.unique(patient_words['patient_id'])
    patient_word_bank = pd.DataFrame()


    for patient in patients:

        cols = ['key_word_1', 'key_word_2', 'key_word_3', 'key_word_4', 'key_word_5']
        patient_words['word_bank'] = patient_words[cols].apply(lambda row: ','.join(row.values.astype(str)), axis=1)
        df = patient_words[['patient_id','word_bank']].drop_duplicates().reset_index()
        patient_word_bank = pd.concat([patient_word_bank,df])
    patient_word_bank = patient_word_bank.drop_duplicates()

            
    patient_word_bank = patient_word_bank.groupby(['patient_id'])['word_bank'].transform(lambda x: ','.join(x)).reset_index().drop_duplicates()
    #patient_word_bank = pd.DataFrame(patient_word_bank).reset_index()

    for i in range(len(patient_word_bank)):
        patient_word_bank['word_bank'][i] = patient_word_bank['word_bank'][i].replace(',', ' ')
        
    pat_id = patient_words.reset_index()
    pat_id = pat_id[['patient_id']]

    patient_word_bank = pd.merge(left=patient_word_bank,right=pat_id,right_index=True,left_index=True,how='left')

    patient_word_bank = patient_word_bank[['patient_id', 'word_bank']].drop_duplicates()

    patient_word_bank     
        
        
