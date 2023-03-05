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

st.set_page_config(page_title='Recommendations, Future Research', layout="centered")
st.sidebar.header('Recommendations, Future Research')

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

with recommendations:
    st.header('Recommendations and Findings')
    
    image4 = Image.open('image12.png')
    st.image(image4, caption='(“Sentiment Classification,” 2021)')
    
    st.markdown(
        """
        - The BERT model is the optimal model and is recommended
            - The BERT topic model can be used to extract the top N keywords
            - These words can be combined to design a word bank
        - Model Best Practices:
            - Word bank quality is higher with more notes offered by the patients
            - Model should be ran as needed when new patients are added to the system
        """)
    
with future_research:
    st.header('Future Research')
    st.markdown(
        """
        - Sentiment analysis using the word banks
            - Retain historical word banks as the model is re-ran
            - Use sentiment analysis to understand if the emotion of the patients is changing over time
        - Enhancing user experience of Streamlit application
            - Adding more user inputs
            - Adding to visual appeal
        """)