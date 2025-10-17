import pandas as pd
import numpy as np
import joblib
import warnings
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import sys

# Append the parent directory to system path to import 01_preprocessing.py
sys.path.append('../')
from src.01_preprocessing import load_and_preprocess_data

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Global Configuration ---
DATA_PATH = '../data/NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv'
MODEL_DIR = '../models/'

# Define filenames for saving artifacts
TFIDF_FILENAME = MODEL_DIR + 'tfidf_vectorizer.joblib'
SCALER_FILENAME = MODEL_DIR + 'scaler_for_engineered_features.joblib'
PCA_FILENAME = MODEL_DIR + 'pca_model.joblib'
KMEANS_FILENAME = MODEL_DIR + 'kmeans_model.joblib'

# --- Model Hyperparameters (Predetermined for demonstration) ---
N_COMPONENTS = 200     # Number of components for PCA
OPTIMAL_K = 8          # Optimal number of clusters determined via Elbow/Silhouette

# --- Features to be scaled (MUST match the order in 01_preprocessing.py) ---
SCALING_FEATURES = ['release_year', 'release_decade', 'is_tv_show', 'is_tv_ma', 'year_added']


# ==============================================================================
# FINAL NLP PROCESSING (Stopwords, Tokenization, Lemmatization)
# ==============================================================================
def final_text_processing(text, lemmatizer, english_stopwords):
    """
    Performs tokenization, stopword removal, and lemmatization.
    """
    if not isinstance(text, str):
        return ""
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Lemmatization and Stopword Removal
    processed_tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token not in english_stopwords and len(token) > 1
    ]
    
    return ' '.join(processed_tokens)


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================
def run_training_pipeline():
    """
    Executes the full model training pipeline: NLP, Vectorization, Scaling, PCA, and K-Means.
    """
    df = load_and_preprocess_data(DATA_PATH)
    if df is None:
        return

    print("\n--- Starting Final NLP & Vectorization ---")
    
    # Initialize NLTK tools
    lemmatizer = WordNetLemmatizer()
    english_stopwords = set(stopwords.words('english'))
    
    # Apply final text processing to cleaned_text_part1
    df['processed_text_final'] = df['cleaned_text_part1'].apply(
        lambda x: final_text_processing(x, lemmatizer, english_stopwords)
    )
    
    # 1. TFIDF Vectorization (Text features)
    print("1. Fitting TFIDF Vectorizer...")
    tfidf = TfidfVectorizer(max_features=5000) 
    tfidf_matrix = tfidf.fit_transform(df['processed_text_final'])
    print(f"   TFIDF Matrix Shape: {tfidf_matrix.shape}")
    
    # 2. Standard Scaling (Numerical features)
    print("2. Fitting StandardScaler...")
    scaler = StandardScaler()
    scaled_engineered_features = scaler.fit_transform(df[SCALING_FEATURES])
    print(f"   Scaled Feature Matrix Shape: {scaled_engineered_features.shape}")
    
    # 3. Combine All Features
    # The full_matrix is typically sparse, we use hstack, but PCA requires dense.
    full_matrix_sparse = hstack([tfidf_matrix, scaled_engineered_features])
    full_matrix_dense = full_matrix_sparse.toarray()
    print(f"3. Combined Full Matrix Shape: {full_matrix_dense.shape}")

    # 4. PCA Dimensionality Reduction
    print(f"4. Fitting PCA (Reducing to {N_COMPONENTS} components)...")
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    pca_matrix = pca.fit_transform(full_matrix_dense)
    
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"   PCA Matrix Shape: {pca_matrix.shape}")
    print(f"   Variance Explained: {explained_variance:.4f}")
    
    # 5. K-Means Clustering
    print(f"5. Fitting K-Means Clustering (K={OPTIMAL_K})...")
    kmeans = KMeans(n_clusters=OPTIMAL_K, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(pca_matrix)
    
    # --- 6. Save Artifacts ---
    print("\n--- Saving Fitted Models and Transformers ---")
    joblib.dump(tfidf, TFIDF_FILENAME)
    print(f"   Saved TFIDF Vectorizer to {TFIDF_FILENAME}")
    
    joblib.dump(scaler, SCALER_FILENAME)
    print(f"   Saved StandardScaler to {SCALER_FILENAME}")
    
    joblib.dump(pca, PCA_FILENAME)
    print(f"   Saved PCA Model to {PCA_FILENAME}")
    
    joblib.dump(kmeans, KMEANS_FILENAME)
    print(f"   Saved K-Means Model to {KMEANS_FILENAME}")

    print("\nTRAINING PIPELINE COMPLETE. All artifacts saved successfully.")


if __name__ == '__main__':
    # Add parent directory to path for module import
    sys.path.append('..')
    
    run_training_pipeline()
