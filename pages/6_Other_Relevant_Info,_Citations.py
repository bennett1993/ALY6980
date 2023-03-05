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

st.set_page_config(page_title='Other Relevant Info, Citations', layout="centered")
st.sidebar.header('Other Relevant Info, Citations')

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
    
with other_relevant:
    st.header('Other Relevant Information')
    
    image5 = Image.open('image13.jpeg')
    st.image(image5, caption='(“Phone Repair,” n.d.)')
    image6 = Image.open('image14.jpeg')
    st.image(image6, caption='(Sullivan, 2020)')
    
    st.markdown(
        """
        - Possible avenues to explore:
            - Phone, watch, and wearable applications for convenient data entry
            - Fingerprint or face login because users often forget their passwords
            - Voice to text in symptom tracker
            - Determining patient emotion from voice
            - Word banks by symptom - common words and phrases used to describe given symptoms
            - Identify commonalities among patients in different groups (gender, age, etc.)
        """)
    
with citations:
    st.header('Citations')
    st.markdown(
        """
        1. Baheti, P. (2023, February 2). A Simple Guide to Data Preprocessing in Machine Learning. V7. https://www.v7labs.com/blog/data-preprocessing-guide
        2. Chenery-Howes, A. (2022, December 21). What Is Latent Semantic Indexing And How Does It Works? Oncrawl - Technical SEO Data. https://www.oncrawl.com/technical-seo/what-is-latent-semantic-indexing/
        3. Grootendorst, M. P. (n.d.). KeyBERT - KeyBERT. https://maartengr.github.io/KeyBERT/api/keybert.html
        4. Mall, R. (2021, December 7). Latent Dirichlet Allocation - Towards Data Science. Medium. https://towardsdatascience.com/latent-dirichlet-allocation-15800c852699
        5. Phone Repair. (n.d.). iFixit. https://www.ifixit.com/Device/Phone
        6. Sentiment Classification Using BERT. (2021, September 8). GeeksforGeeks. https://www.geeksforgeeks.org/sentiment-classification-using-bert/
        7. Sullivan, F. &. (2020, September 29). Wearable Technologies and Healthcare: Differentiating the Toys and Tools for Quantified-Self' with Actionable Health Use Cases. Frost & Sullivan. https://www.frost.com/frost-perspectives/wearable-technologies-and-healthcare-differentiating-toys-and-tools-quantified-self-actionable-health-use-cases/
        8. Zhu, R., Tu, X., & Huang, J. X. (2021). Utilizing BERT for biomedical and clinical text mining. Data Analytics in Biomedical Engineering and Healthcare, 73-103. https://doi.org/10.1016/b978-0-12-819314-3.00005-7
        """)