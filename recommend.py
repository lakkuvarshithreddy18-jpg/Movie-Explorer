import pickle
import sys
from sklearn.metrics.pairwise import linear_kernel

def load_models():
    """Load precomputed model files and metadata."""
    print("Loading indices...")
    with open("indices.pkl", "rb") as f:
        indices = pickle.load(f)

    print("Loading TF-IDF matrix...")
    with open("tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    print("Loading movie dataset...")
    with open("df.pkl", "rb") as f:
        df = pickle.load(f)

    return indices, tfidf_matrix, df

def get_recommendations(title, indices, tfidf_matrix, df, top_n=10):
    """
    Get top N movie recommendations based on cosine similarity of TF-IDF vectors.
    """
    if title not in indices:
        # Case-insensitive search fallback
        matching_titles = [t for t in indices.keys() if str(title).lower() in str(t).lower()]
        if not matching_titles:
            return f"Movie '{title}' not found in database."
        title = matching_titles[0]
        print(f"Using closest match: '{title}'")

    idx = indices[title]

    # Handle duplicate titles where index could be a Series or list
    if hasattr(idx, '__iter__') and not isinstance(idx, int):
        idx = int(idx.iloc[0] if hasattr(idx, 'iloc') else idx[0])

    # Calculate cosine similarity with all movie vectors
    cosine_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Get the indices of the highest similarity scores excluding queried movie
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    movie_indices = []
    seen_titles = {str(title).lower()}
    for i, _ in sim_scores:
        if i == idx:
            continue
        t = str(df.iloc[i]['title']) if 'title' in df.columns else None
        if t and t.lower() in seen_titles:
            continue
        if t:
            seen_titles.add(t.lower())
        movie_indices.append(i)
        if len(movie_indices) >= top_n:
            break

    return df[['title']].iloc[movie_indices] if 'title' in df.columns else df.iloc[movie_indices]


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "Toy Story"
    print(f"Fetching recommendations for: '{query}'\n")
    try:
        indices, tfidf_matrix, df = load_models()
        recs = get_recommendations(query, indices, tfidf_matrix, df)
        print("\nTop Recommendations:")
        print(recs)
    except Exception as e:
        print(f"Error: {e}")
