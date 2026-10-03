# 🎬 Movie Explorer

A content-based movie recommendation system that suggests similar movies using Natural Language Processing (NLP) with TF-IDF vectorization, Cosine Similarity, and optional TMDB API integration.

---

## 📌 Project Overview

Movie Explorer analyzes movie overviews, taglines, and metadata to identify the most relevant and stylistically similar movies to a given title.

### Key Components
- **`app.py`**: Streamlit interactive web application ("MovieMate") with movie browsing, search, genres, and TF-IDF recommendations.
- **`main.py`**: FastAPI backend service exposing endpoints for movie recommendations, search bundles, and TMDB integration.
- **`recommend.py`**: Python CLI script to query and fetch top recommendations directly in terminal.
- **`df.pkl`**: Preprocessed DataFrame containing 45,000+ movie titles, metadata, and overviews.
- **`indices.pkl`**: Fast lookup dictionary mapping movie titles to index positions.
- **`tfidf.pkl`**: Trained `TfidfVectorizer` model.
- **`tfidf_matrix.pkl`**: Precomputed sparse TF-IDF matrix representation of all movie descriptions.

---

## 🚀 Getting Started

### 1. Prerequisites & Installation
Ensure you have Python 3.10+ installed. Activate your virtual environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
If you wish to use TMDB poster images and genre exploration features, add your free TMDB API key to `.env`:

```env
TMDB_API_KEY=your_tmdb_api_key_here
```
*(The local TF-IDF recommendation engine works fully without a TMDB key).*

---

## 🏃 Running the Application

### Option A: Run the Streamlit Web Application
Start the interactive UI:

```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

### Option B: Run the FastAPI Server
Start the API server with live reloading:

```bash
python main.py
```
Or via Uvicorn:
```bash
uvicorn main:app --reload
```

- **Interactive API Docs (Swagger UI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- **TF-IDF Recommendations Endpoint**: `GET /recommend/tfidf?title=Avatar&top_n=5`
- **Full Movie Search Bundle**: `GET /movie/search?query=Inception`

---

### Option C: Run CLI Recommendations

You can also run recommendations directly from the command line:

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
