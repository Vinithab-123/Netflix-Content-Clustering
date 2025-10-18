# Netflix Content Taxonomy & Recommendation Clustering

## Project Type
Unsupervised Machine Learning: Clustering/Content Taxonomy

## Problem Statement
The challenge is to create an objective, scalable content taxonomy for the growing Netflix catalog by identifying distinct content niches (clusters) to enhance recommendation systems and inform content acquisition strategy.

## Project Summary
This project utilizes unsupervised machine learning (K-Means and DBSCAN) on the Netflix content catalog to develop a data-driven content taxonomy. It involves extensive data preprocessing, feature engineering, text vectorization (TFIDF), and dimensionality reduction (PCA) to transform raw title metadata into a compact, meaningful feature matrix. The final model assigns each movie and TV show to a specific cluster, effectively segmenting the catalog based on intrinsic semantic and metadata features.



## Conclusion
The primary goal of this project—to segment the Netflix catalog based on intrinsic content characteristics—was successfully achieved using an unsupervised machine learning pipeline. The final model effectively created a content taxonomy, grouping titles based on their semantic similarity (driven by text features) and metadata (driven by engineered features).



## Key Files & Code Steps

1.  **Preprocessing (`src/01_preprocessing.py`):** Handling missing values, outlier treatment, and categorical/text encoding (TFIDF).
2.  **Modeling (`src/02_model_training.py`):** Feature Scaling, Dimensionality Reduction (PCA), and K-Means clustering with optimal K determination.
3.  **Prediction (`src/03_prediction_pipeline.py`):** Demonstration of scoring a new, unseen Netflix title.

## Technologies Used
- Python
- Pandas, NumPy, SciPy
- Scikit-learn (K-Means, PCA, StandardScaler, TFIDF)
- NLTK (Lemmatization, Stopwords)
