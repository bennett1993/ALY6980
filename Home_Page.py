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

st.set_page_config(page_title='Home Page', layout="centered")
st.sidebar.header('Home Page')

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

with header:
    st.title('ALY 6980 XN Project')
    st.write("Fledgling's Flight")
    st.write('Bennett Furman, Samuel Lurie, Srivaths Ganesh Rao Mahadikar')
    st.write('Welcome to our Streamlit Application!')
    st.write('This Streamlit application is linked to this GitHub repository: https://github.com/bennett1993/ALY6980')
    
    image = Image.open('image15.png')
    st.image(image)