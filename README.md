# Netflix Content Taxonomy & Recommendation Clustering

## Project Type
Unsupervised Machine Learning: Clustering/Content Taxonomy

## Problem Statement
The challenge is to create an objective, scalable content taxonomy for the growing Netflix catalog by identifying distinct content niches (clusters) to enhance recommendation systems and inform content acquisition strategy.

## Project Summary
[Insert the Project Summary we wrote previously.]

---

## Conclusion
[Insert the full Conclusion we wrote previously.]

---

## Key Files & Code Steps

1.  **Preprocessing (`src/01_preprocessing.py`):** Handling missing values, outlier treatment, and categorical/text encoding (TFIDF).
2.  **Modeling (`src/02_model_training.py`):** Feature Scaling, Dimensionality Reduction (PCA), and K-Means clustering with optimal K determination.
3.  **Prediction (`src/03_prediction_pipeline.py`):** Demonstration of scoring a new, unseen Netflix title.

## Technologies Used
- Python
- Pandas, NumPy, SciPy
- Scikit-learn (K-Means, PCA, StandardScaler, TFIDF)
- NLTK (Lemmatization, Stopwords)