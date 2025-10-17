import numpy as np
import joblib
import re
import warnings
from scipy.sparse import hstack
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Suppress warnings
warnings.filterwarnings('ignore')

# --- File Paths ---
MODEL_DIR = '../models/'
TFIDF_FILENAME = MODEL_DIR + 'tfidf_vectorizer.joblib'
SCALER_FILENAME = MODEL_DIR + 'scaler_for_engineered_features.joblib'
PCA_FILENAME = MODEL_DIR + 'pca_model.joblib'
KMEANS_FILENAME = MODEL_DIR + 'kmeans_model.joblib'

# --- NLP Constants and Functions (Replicated from 01/02 for self-containment) ---
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
    pattern = re.compile('({})'.format('|'.join(re.escape(key) for key in mapping.keys())), re.IGNORECASE)
    def replace(match): return mapping[match.group(0).lower()]
    return pattern.sub(replace, text)

def initial_clean_text(text):
    if not isinstance(text, str): return ""
    text = expand_contractions(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    words = [word for word in text.split() if not re.search(r'\d', word)]
    text = ' '.join(words)
    return re.sub(r'\s+', ' ', text).strip()

# NOTE: The TFIDF model inherently includes Stopword/Lemmatization logic.
# For new prediction, we only need the clean text for the TFIDF transform.

# ==============================================================================
# MOCKING CLASSES (in case joblib files are missing)
# ==============================================================================
class MockTransformer:
    def transform(self, X): 
        if hasattr(X, 'toarray'): X = X.toarray()
        if X.shape[1] > 5: return X[:, :200] # Mock PCA reduction
        return X
    def predict(self, X): return np.array([3]) 

def load_models_or_mock():
    """Loads saved models, or initializes mocks if files are not found."""
    try:
        loaded_kmeans = joblib.load(KMEANS_FILENAME)
        loaded_pca = joblib.load(PCA_FILENAME)
        loaded_tfidf = joblib.load(TFIDF_FILENAME)
        loaded_scaler = joblib.load(SCALER_FILENAME)
        print("SUCCESS: All models and transformers loaded.")
        return loaded_kmeans, loaded_pca, loaded_tfidf, loaded_scaler
    except FileNotFoundError:
        print("WARNING: Model files not found. Using MOCK objects for demonstration.")
        # Initialize Mocks (must be fitted to handle transform/predict calls)
        loaded_kmeans = MockTransformer()
        loaded_pca = MockTransformer() 
        loaded_tfidf = TfidfVectorizer(max_features=5000)
        # Mock fit TFIDF to avoid NotFittedError
        loaded_tfidf.fit(["scifi thriller action rogue ai distant colony", "comedy romance director cast family fun"])
        loaded_scaler = StandardScaler()
        loaded_scaler.n_features_in_ = 5 
        loaded_scaler.mean_ = np.zeros(5) 
        loaded_scaler.scale_ = np.ones(5)
        return loaded_kmeans, loaded_pca, loaded_tfidf, loaded_scaler


# ==============================================================================
# UNSEEN DATA POINT
# ==============================================================================
unseen_title_data = {
    'title': 'New Sci-Fi Thriller',
    'director': 'Jane Doe',
    'cast': 'A-List Actor, B-List Actress',
    'listed_in': 'Sci-Fi, Thrillers, Action',
    'description': 'A gritty, futuristic chase against a rogue AI on a distant colony.',
    # Engineered features (must match SCALING_FEATURES order: year, decade, is_show, is_ma, year_added)
    'release_year': 2024,
    'release_decade': 2020, 
    'is_tv_show': 0, # Movie
    'is_tv_ma': 1,   # TV-MA rating
    'year_added': 2024      
}
SCALING_FEATURES = ['release_year', 'release_decade', 'is_tv_show', 'is_tv_ma', 'year_added']


# ==============================================================================
# PREDICTION FUNCTION
# ==============================================================================
def predict_cluster_for_title(title_data, kmeans_model, pca_model, tfidf_model, scaler_model):
    """Runs the unseen data through the full transformation pipeline."""
    
    # --- 1. Text Preprocessing ---
    unseen_text = (
        title_data['title'] + ' ' + title_data['director'] + ' ' + title_data['cast'] + ' ' +
        title_data['listed_in'] + ' ' + title_data['description']
    )
    cleaned_unseen_text = initial_clean_text(unseen_text)
    
    # 2. Vectorize the text using the saved TFIDF transformer
    unseen_tfidf = tfidf_model.transform([cleaned_unseen_text])

    # 3. Create and Scale the other features
    unseen_engineered_features = np.array([title_data[f] for f in SCALING_FEATURES]).reshape(1, -1)
    
    # Apply the SAVED StandardScaler
    unseen_scaled_features = scaler_model.transform(unseen_engineered_features)

    # 4. Combine features (unseen_full_matrix is sparse)
    unseen_full_matrix = hstack([unseen_tfidf, unseen_scaled_features])

    # 5. Apply the SAVED PCA reduction
    unseen_pca_matrix = pca_model.transform(unseen_full_matrix)

    # 6. Predict the Cluster
    predicted_cluster = kmeans_model.predict(unseen_pca_matrix)
    
    return predicted_cluster[0]


if __name__ == '__main__':
    # Load or initialize models
    kmeans, pca, tfidf, scaler = load_models_or_mock()
    
    print("\n--- Running Prediction Pipeline ---")
    
    # Predict the cluster
    predicted_id = predict_cluster_for_title(
        unseen_title_data, kmeans, pca, tfidf, scaler
    )

    print("\n=====================================================")
    print(f"New Title: '{unseen_title_data['title']}'")
    print(f"The model predicts this title belongs to **Cluster: {predicted_id}**")
    print("=====================================================")
