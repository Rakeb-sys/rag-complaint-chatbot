import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from IPython.display import display

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import nltk

nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def set_aesthetics(palette_style="muted", font_scale=1.1, figure_figsize=(12, 5)):
    """Sets the default plotting theme and options for EDA."""
    sns.set_theme(style="whitegrid", palette=palette_style, font_scale=font_scale)
    plt.rcParams.update({
        "figure.figsize": figure_figsize,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
    })
    pd.set_option("display.float_format", "{:.4f}".format)
    print("Libraries loaded & global aesthetics set ✓")

def load_data(file_path):
    """Loads a dataset from CSV and prints basic shape and memory information."""
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory : {df.memory_usage(deep=True).sum()/1e6:.1f} MB")
    return df

def assess_quality(df):
    """Reports missing values in the DataFrame."""
    # Missing values
    miss = df.isnull().sum()
    miss_pct = (miss / len(df) * 100).round(4)
    quality = pd.DataFrame({"missing_count": miss, "missing_%": miss_pct})
    quality = quality[quality.missing_count > 0]
    
    if quality.empty:
        print("No missing values detected.")
    else:
        print("Columns with missing values:")
        display(quality)
        
def clean_text_noise(text):
    text = str(text).lower()
    
    # STEP 1: Remove URLs first
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # STEP 2: Remove Phone Numbers (while the digits are still grouped)
    # This pattern looks for 7 to 10 digits in a row
    text = re.sub(r'\b\d{3}[-.\s]?\d{4}\b', '', text) # 7 digit format
    text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '', text) # 10 digit format
    text = re.sub(r'\b[A-Z]{2}\d{10,}\b', '', text, flags=re.I)   # Simple IBAN/IDs
    # STEP 3: Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # STEP 4: remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    return text.strip() 

def normalize_text(text):
    tokens = word_tokenize(text)
    
    # 1. Remove stopwords
    tokens = [t for t in tokens if t not in stop_words]
    
    # 2. Apply Lemmatization for BOTH Nouns and Verbs
    # Notice we call .lemmatize(word, pos='v') to target verbs!
    lemmas = [lemmatizer.lemmatize(t, pos='v') for t in tokens]
    
    # 3. Apply it again for Nouns (default)
    lemmas = [lemmatizer.lemmatize(t, pos='n') for t in lemmas]
    
    return " ".join(lemmas)


