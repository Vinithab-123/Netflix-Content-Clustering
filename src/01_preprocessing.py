import pandas as pd
import numpy as np
import re
import nltk
import warnings
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Download necessary NLTK packages if not already present (run this once)
# try:
#     nltk.data.find('corpora/stopwords')
# except nltk.downloader.DownloadError:
#     nltk.download('stopwords')
# try:
#     nltk.data.find('corpora/wordnet')
# except nltk.downloader.DownloadError:
#     nltk.download('wordnet')
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     nltk.download('punkt')

# ==============================================================================
# CONTRACTION HANDLING
# ==============================================================================
CONTRACTION_MAP = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
    "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
    "i'm": "i am", "isn't": "is not", "it's": "it is", "mightn't": "might not",
    "mustn't": "must not", "shan't": "shall not", "she's": "she is", "shouldn't": "should not",
    "that's": "that is", "there's": "there is", "they're": "they are", "wasn't": "was not",
    "we're": "we are", "weren't": "were not", "what's": "what is", "where's": "where is",
    "won't": "will not", "wouldn't": "would not", "you're": "you are"
}

def expand_contractions(text, mapping=CONTRACTION_MAP):
    """Replaces contractions in string with their expanded form."""
    pattern = re.compile('({})'.format('|'.join(re.escape(key) for key in mapping.keys())), re.IGNORECASE)
    def replace(match):
        return mapping[match.group(0).lower()]
    return pattern.sub(replace, text)

# ==============================================================================
# INITIAL TEXT CLEANING FUNCTION (Applied before NLTK operations)
# ==============================================================================
def initial_clean_text(text):
    """
    Performs the initial text cleaning steps:
    1. Contraction expansion.
    2. Lowercasing.
    3. URL removal.
    4. Punctuation removal.
    5. Removal of words containing digits.
    """
    if not isinstance(text, str):
        return ""

    # 1. Expand Contraction
    text = expand_contractions(text)

    # 2. Lower Casing
    text = text.lower()

    # 3. Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # 4. Remove Punctuations (replace with space to prevent words from joining)
    text = re.sub(r'[^\w\s]', ' ', text)

    # 5. Remove words containing digits
    words = text.split()
    words = [word for word in words if not re.search(r'\d', word)]
    text = ' '.join(words)

    # Final step: Remove extra white spaces
    return re.sub(r'\s+', ' ', text).strip()

# ==============================================================================
# MAIN PREPROCESSING PIPELINE
# ==============================================================================
def load_and_preprocess_data(file_path):
    """
    Loads data, imputes missing values, performs feature engineering,
    and executes initial text cleaning.
    """
    print(f"Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Exiting.")
        return None

    # --- 1. Missing Value Imputation and Date Conversion ---
    
    # Fill text columns with 'Missing' or mode
    for col in ['director', 'cast', 'country', 'rating', 'description', 'listed_in']:
        df[col] = df[col].fillna('Missing')
        
    # Handle 'date_added': Impute and Convert to datetime (CRITICAL FIX)
    df['date_added'] = df['date_added'].fillna(df['date_added'].mode()[0])
    # The 'format='mixed'' argument handles various date formats efficiently
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce', format='mixed')
    
    # Handle 'release_year'
    df['release_year'] = df['release_year'].fillna(df['release_year'].median()).astype(int)
    
    # --- 2. Feature Engineering ---
    
    # Feature 1: Is it a TV Show? (Binary)
    df['is_tv_show'] = np.where(df['type'] == 'TV Show', 1, 0)
    
    # Feature 2: Release Decade
    df['release_decade'] = (df['release_year'] // 10 * 10).astype(int)
    
    # Feature 3: Year Added to Netflix
    df['year_added'] = df['date_added'].dt.year.astype(int)
    
    # Feature 4: Is it a TV-MA rating? (High-risk content flag)
    df['is_tv_ma'] = np.where(df['rating'] == 'TV-MA', 1, 0)
    
    # --- 3. Text Feature Combination and Cleaning ---
    
    # Combine all relevant text/categorical features for clustering
    df['cluster_features'] = (
        df['title'] + ' ' + df['director'] + ' ' + df['cast'] + ' ' +
        df['listed_in'] + ' ' + df['description']
    )
    
    # Apply initial cleaning function
    df['cleaned_text_part1'] = df['cluster_features'].apply(initial_clean_text)
    
    print("Preprocessing complete. DataFrame ready for NLTK/Vectorization.")
    return df

if __name__ == '__main__':
    # Adjust the file path to match the directory structure
    DATA_PATH = '../data/NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv'
    
    processed_df = load_and_preprocess_data(DATA_PATH)
    
    if processed_df is not None:
        print("\nProcessed Data Head:")
        print(processed_df[['show_id', 'type', 'release_year', 'year_added', 'is_tv_show', 'cleaned_text_part1']].head())
        print(f"\nDataFrame shape: {processed_df.shape}")