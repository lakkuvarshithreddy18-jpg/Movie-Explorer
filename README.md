# 🎬 Movie Explorer

A content-based movie recommendation system that suggests similar movies using Natural Language Processing (NLP) with TF-IDF vectorization and Cosine Similarity.

---

## 📌 Project Overview

Movie Explorer analyzes movie overviews, taglines, and metadata to identify the most relevant and stylistically similar movies to a given title.

### Key Components
- **`df.pkl`**: Preprocessed DataFrame containing movie titles, metadata, and overviews.
- **`indices.pkl`**: Fast lookup dictionary mapping movie titles to index positions.
- **`tfidf.pkl`**: Trained `TfidfVectorizer` model.
- **`tfidf_matrix.pkl`**: Precomputed sparse TF-IDF matrix representation of all movie descriptions.
- **`recommend.py`**: Python script to query and fetch top recommendations.

---

## 🚀 Getting Started

### 1. Prerequisites & Installation
Ensure you have Python 3.10+ installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run Recommendations
You can get recommendations by running:

```bash
python recommend.py "The Dark Knight"
```

Or run without arguments to test with the default title (`Toy Story`):

```bash
python recommend.py
```

---

## ⚙️ How It Works

1. **TF-IDF Vectorization**: Transforms unstructured movie text descriptions into numerical vector embeddings capturing keyword significance.
2. **Cosine Similarity**: Calculates pairwise similarity between movies using dot products of their TF-IDF vector representations.
3. **Ranking**: Orders similarity scores and returns the top matching titles.
