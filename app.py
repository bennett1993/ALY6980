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

header = st.container()
dataset = st.container()
model_training = st.container()

with header:
    st.title('ALY 6080 XN Project')
    st.text('TBI Patient Keyword Extraction')


with dataset:
    st.header('Power of Patients Text Data')
    notes = pd.read_csv(r"C:\Users\Bennett\Documents\NEU\2023\Winter P1\ALY 6080\Streamlit App\Data\additional_notes.csv")
    st.write(notes)
    
with model_training:
    st.header('Time to train the model!')
    st.text('Here you get to choose the parameters of the keyword extration model')