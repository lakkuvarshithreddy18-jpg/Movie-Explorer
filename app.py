import streamlit as st
import requests
import os
import re
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from dotenv import load_dotenv

# Import recommendation engine and datasets
import main
from main import tfidf_recommend_titles
from blend_recommend import blend_recommend

load_dotenv(override=True)

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="MovieMate - Movie Explorer & Recommender",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# TMDB CONFIG
# -----------------------------

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip() or None
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_URL = "https://image.tmdb.org/t/p/w500"

# Pre-load models at app launch
main.ensure_models_loaded()


# -----------------------------
# TMDB POSTER HELPER
# -----------------------------

@st.cache_data(show_spinner=False)
def get_poster_url(title, year=None):
    """
    Fetch movie poster URL from TMDB search or movie details.
    Cached via st.cache_data. Returns full image URL or None on any failure.
    """
    api_key = os.getenv("TMDB_API_KEY", "").strip()
    if not api_key:
        return None

    clean_title = str(title).strip()
    if not clean_title:
        return None

    # Auto-extract year if present in title like "Movie (1999)" and year not provided
    if year is None:
        m = re.search(r"\((\d{4})\)$", clean_title)
        if m:
            year = m.group(1)
            clean_title = clean_title[:m.start()].strip()

    # Check if dataset has tmdb id or movie id column
    df = main.get_df() if hasattr(main, "get_df") else None
    has_id_col = False
    id_col_name = None
    if df is not None:
        for col in ["tmdb_id", "id", "movie_id"]:
            if col in df.columns:
                has_id_col = True
                id_col_name = col
                break

    try:
        if has_id_col and id_col_name:
            match = df[df["title"].astype(str).str.lower() == clean_title.lower()]
            if not match.empty and pd.notna(match.iloc[0][id_col_name]):
                m_id = int(match.iloc[0][id_col_name])
                resp = requests.get(
                    f"https://api.themoviedb.org/3/movie/{m_id}",
                    params={"api_key": api_key},
                    timeout=5
                )
                if resp.status_code == 200:
                    data = resp.json()
                    poster_path = data.get("poster_path")
                    if poster_path:
                        return f"https://image.tmdb.org/t/p/w500{poster_path}"

        # Search endpoint fallback / default
        params = {
            "api_key": api_key,
            "query": clean_title,
            "language": "en-US",
        }
        if year:
            params["year"] = str(year).strip()
            params["primary_release_year"] = str(year).strip()

        resp = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params=params,
            timeout=5
        )
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results")
            if results and len(results) > 0:
                poster_path = results[0].get("poster_path")
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500{poster_path}"
        return None
    except Exception:
        return None


def fetch_posters_parallel(items):
    """
    Fetch poster URLs in parallel using ThreadPoolExecutor for a list of items.
    Items can be movie dicts, (title, score) tuples, or title strings.
    """
    if not items:
        return []

    def _fetch(item):
        if isinstance(item, dict):
            t = item.get("title", "")
            rel = item.get("release_date", "")
            y = rel[:4] if rel and len(rel) >= 4 else None
            return get_poster_url(t, year=y)
        elif isinstance(item, tuple):
            t = item[0]
            return get_poster_url(t)
        else:
            return get_poster_url(str(item))

    with ThreadPoolExecutor(max_workers=min(len(items), 10)) as executor:
        return list(executor.map(_fetch, items))

# -----------------------------
# CUSTOM CSS
# -----------------------------

st.markdown("""
<style>
.main {
    background-color: #0f1117;
}
.movie-card {
    background-color: #1b1d25;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 15px;
}
.movie-title {
    font-size: 20px;
    font-weight: bold;
}
.movie-info {
    color: #aaaaaa;
}
h1 {
    text-align: center;
}
.poster-placeholder {
    background: linear-gradient(135deg, #1e222d 0%, #2a2f3d 100%);
    border-radius: 8px;
    height: 220px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: #8892b0;
    margin-bottom: 8px;
    border: 1px solid #323b4e;
}
.stImage img {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# DATA HELPERS & TMDB FUNCTIONS
# -----------------------------

def tmdb_request(endpoint, params=None):
    api_key = os.getenv("TMDB_API_KEY", "").strip() or TMDB_API_KEY
    if not api_key:
        return None
    if params is None:
        params = {}
    params["api_key"] = api_key
    try:
        response = requests.get(
            BASE_URL + endpoint,
            params=params,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def get_local_movie(title):
    """Retrieve movie details from the local 45k DataFrame."""
    df = main.get_df()
    title_to_idx = main.get_title_to_idx()
    key = str(title).strip().lower()
    idx = None
    if title_to_idx and key in title_to_idx:
        idx = title_to_idx[key]
    elif title_to_idx:
        for k, i in title_to_idx.items():
            if key in k or (len(key) >= 3 and len(k) >= 3 and k in key):
                idx = i
                break

    if idx is not None and df is not None and 0 <= idx < len(df):
        row = df.iloc[idx]
        return {
            "id": int(idx),
            "title": str(row["title"]),
            "vote_average": float(row["vote_average"]) if pd.notna(row.get("vote_average")) else 0.0,
            "release_date": "",
            "overview": str(row.get("overview", "No local overview available.")),
            "poster_path": None,
        }
    return None


def search_movie(title):
    """Search for a single movie using TMDB first, falling back to local dataset."""
    clean_title = str(title).strip()
    api_key = os.getenv("TMDB_API_KEY", "").strip() or TMDB_API_KEY
    if api_key:
        data = tmdb_request(
            "/search/movie",
            {
                "query": clean_title,
                "language": "en-US"
            }
        )
        if data and data.get("results"):
            return data["results"][0]
    return get_local_movie(clean_title)


def search_movies_multi(query):
    """Search for matching movies using TMDB with local dataset fallback."""
    clean_query = str(query).strip()
    api_key = os.getenv("TMDB_API_KEY", "").strip() or TMDB_API_KEY
    if api_key:
        data = tmdb_request(
            "/search/movie",
            {
                "query": clean_query,
                "language": "en-US",
                "page": 1
            }
        )
        if data and data.get("results"):
            return data["results"]

    # Fallback to local DataFrame
    df = main.get_df()
    if df is not None:
        matched = df[df["title"].astype(str).str.contains(clean_query, case=False, na=False)].head(15)
        results = []
        for idx, row in matched.iterrows():
            results.append({
                "id": int(idx),
                "title": str(row["title"]),
                "vote_average": float(row["vote_average"]) if pd.notna(row.get("vote_average")) else 0.0,
                "release_date": "",
                "overview": str(row.get("overview", "")),
                "poster_path": None,
            })
        return results
    return []


def get_popular_movies():
    """Fetch popular movies via TMDB or top ranked movies from local dataset."""
    api_key = os.getenv("TMDB_API_KEY", "").strip() or TMDB_API_KEY
    if api_key:
        data = tmdb_request(
            "/movie/popular",
            {
                "language": "en-US",
                "page": 1
            }
        )
        if data and data.get("results"):
            return data.get("results", [])

    # Local fallback sorted by popularity
    df = main.get_df()
    if df is not None:
        pop = pd.to_numeric(df.get("popularity"), errors="coerce").fillna(0)
        top_indices = pop.sort_values(ascending=False).index[:15]
        results = []
        for idx in top_indices:
            row = df.iloc[idx]
            results.append({
                "id": int(idx),
                "title": str(row["title"]),
                "vote_average": float(row["vote_average"]) if pd.notna(row.get("vote_average")) else 0.0,
                "release_date": "",
                "overview": str(row.get("overview", "")),
                "poster_path": None,
            })
        return results
    return []


def get_movies_by_genre(genre_name, genre_id):
    """Fetch movies by genre from TMDB or local dataset."""
    api_key = os.getenv("TMDB_API_KEY", "").strip() or TMDB_API_KEY
    if api_key:
        data = tmdb_request(
            "/discover/movie",
            {
                "with_genres": genre_id,
                "sort_by": "popularity.desc",
                "language": "en-US"
            }
        )
        if data and data.get("results"):
            return data.get("results", [])

    # Fallback to local dataset by genre text matching
    df = main.get_df()
    if df is not None:
        matched = df[df["genres"].astype(str).str.contains(genre_name, case=False, na=False)].head(15)
        results = []
        for idx, row in matched.iterrows():
            results.append({
                "id": int(idx),
                "title": str(row["title"]),
                "vote_average": float(row["vote_average"]) if pd.notna(row.get("vote_average")) else 0.0,
                "release_date": "",
                "overview": str(row.get("overview", "")),
                "poster_path": None,
            })
        return results
    return []



# -----------------------------
# MOVIE CARD DISPLAY
# -----------------------------

def display_movie(movie, poster_url=None):
    if not poster_url:
        poster_url = movie.get("poster_url")
    if not poster_url and movie.get("poster_path"):
        poster_url = IMAGE_URL + movie["poster_path"]
    if not poster_url and movie.get("title"):
        rel = movie.get("release_date", "")
        y = rel[:4] if rel and len(rel) >= 4 else None
        poster_url = get_poster_url(movie["title"], year=y)

    if poster_url:
        st.image(
            poster_url,
            use_container_width=True
        )
    else:
        st.markdown(
            f"""
            <div class="poster-placeholder">
                <div style="font-size: 42px;">🎬</div>
                <div style="font-size: 13px; font-weight: 500;">Movie Explorer</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        f"**{movie.get('title', 'Unknown')}**"
    )

    rating = movie.get("vote_average")
    if rating is not None and isinstance(rating, (int, float)):
        st.write(f"⭐ {rating:.1f}/10")
    else:
        st.write("⭐ N/A")

    release = movie.get("release_date", "")
    if release:
        st.write(f"📅 {release}")


# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("🎬 MovieMate")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔎 Search",
        "🎯 Recommendations",
        "🎭 Categories"
    ]
)

current_key = os.getenv("TMDB_API_KEY", "").strip() or TMDB_API_KEY
if not current_key:
    st.sidebar.info("💡 **Local Mode Active**\n\n45,000+ movies indexed locally. To enable online movie posters and live trailers, add your free key to `TMDB_API_KEY` in `.env`.")
else:
    st.sidebar.success("⚡ Online mode active")


# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.title("🎬 MovieMate")

    st.subheader(
        "Discover movies you will love"
    )

    st.write(
        "Search for movies, explore popular titles "
        "and get personalized recommendations powered by TF-IDF & Cosine Similarity."
    )

    st.divider()

    st.subheader("🔥 Popular Movies")

    movies = get_popular_movies()

    if movies:

        poster_urls = fetch_posters_parallel(movies[:10])
        cols = st.columns(5)

        for i, (movie, p_url) in enumerate(zip(movies[:10], poster_urls)):

            with cols[i % 5]:

                display_movie(movie, poster_url=p_url)

                if st.button(
                    "View Details",
                    key=f"home_{movie.get('id', i)}"
                ):

                    st.session_state["selected_movie"] = movie
                    st.rerun()


# =========================================================
# SEARCH PAGE
# =========================================================

elif page == "🔎 Search":

    st.title("🔎 Search Movies")

    query = st.text_input(
        "Enter movie name",
        placeholder="Example: Avengers, Inception, Toy Story"
    )

    if query:

        movies = search_movies_multi(query)

        if movies:

            st.subheader(
                f"Search results for: '{query}'"
            )

            poster_urls = fetch_posters_parallel(movies[:10])
            cols = st.columns(5)

            for i, (movie, p_url) in enumerate(zip(movies[:10], poster_urls)):

                with cols[i % 5]:

                    display_movie(movie, poster_url=p_url)

                    if st.button(
                        "View Details",
                        key=f"search_{movie.get('id', i)}"
                    ):

                        st.session_state["selected_movie"] = movie
                        st.rerun()
        else:
            st.warning(f"No movies found matching '{query}'.")


# =========================================================
# RECOMMENDATION PAGE
# =========================================================

elif page == "🎯 Recommendations":

    st.title("🎯 Movie Recommendations")

    st.write(
        "Enter a movie you like and our Natural Language TF-IDF engine "
        "will find stylistically and thematically similar movies."
    )

    query = st.text_input(
        "Enter a movie name",
        placeholder="Example: Inception, The Dark Knight, Toy Story"
    )

    top_n = st.slider(
        "Number of recommendations",
        5,
        20,
        10
    )

    if st.button("🚀 Get Recommendations"):

        if query:

            try:

                recommendations = tfidf_recommend_titles(
                    query,
                    top_n=top_n
                )

                if recommendations:

                    st.subheader(
                        f"🎬 Movies Similar to '{query}'"
                    )

                    poster_urls = fetch_posters_parallel(recommendations)

                    for item, p_url in zip(recommendations, poster_urls):

                        if isinstance(item, tuple):
                            title_str, score = item[0], item[1]
                        else:
                            title_str, score = str(item), None

                        movie = search_movie(title_str)
                        if not movie:
                            movie = {
                                "id": hash(title_str),
                                "title": title_str,
                                "vote_average": "N/A",
                                "release_date": "",
                                "overview": "Similarity recommendation from NLP model.",
                                "poster_path": None,
                            }

                        if not p_url and movie.get("poster_path"):
                            p_url = IMAGE_URL + movie["poster_path"]

                        col1, col2 = st.columns([1, 4])

                        with col1:

                            if p_url:
                                st.image(
                                    p_url,
                                    use_container_width=True
                                )
                            else:
                                st.markdown(
                                    f"""
                                    <div class="poster-placeholder" style="height: 160px;">
                                        <div style="font-size: 32px;">🎬</div>
                                        <div style="font-size: 11px;">Movie Explorer</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                        with col2:

                            st.subheader(
                                movie.get("title", title_str)
                            )

                            rating_str = movie.get("vote_average", "N/A")
                            rating_disp = f"{rating_str:.1f}/10" if isinstance(rating_str, (int, float)) else str(rating_str)
                            score_disp = f" | 🎯 Similarity Match: {score:.1%}" if score is not None else ""
                            st.write(f"⭐ Rating: {rating_disp}{score_disp}")

                            if movie.get("release_date"):
                                st.write(f"📅 Release: {movie['release_date']}")

                            overview_text = movie.get("overview") or "No description available."
                            st.write(f"📝 {overview_text}")

                        st.divider()

                else:

                    st.warning(
                        "No recommendations found."
                    )

            except Exception as e:

                st.error(
                    f"Recommendation error: {e}"
                )

        else:

            st.warning(
                "Please enter a movie name."
            )

    # =========================================================
    # BLENDED RECOMMENDATIONS  (new section)
    # =========================================================

    st.divider()

    st.subheader("Recommend from Two Movies")

    blend_col1, blend_col2 = st.columns(2)
    with blend_col1:
        blend_title_a = st.text_input(
            "First movie",
            key="blend_first_movie"
        )
    with blend_col2:
        blend_title_b = st.text_input(
            "Second movie",
            key="blend_second_movie"
        )

    if st.button("Get Blended Recommendations", key="blend_recommendations_button"):
        if not blend_title_a.strip() and not blend_title_b.strip():
            st.error("Please enter both movie names.")
        elif not blend_title_a.strip():
            st.error("Please enter the first movie name.")
        elif not blend_title_b.strip():
            st.error("Please enter the second movie name.")
        else:
            try:
                mode_label, blend_results = blend_recommend(
                    blend_title_a.strip(),
                    blend_title_b.strip(),
                    top_n=10,
                )

                if blend_results:
                    st.info(mode_label)

                    poster_urls = fetch_posters_parallel(blend_results)

                    for item, p_url in zip(blend_results, poster_urls):
                        title_str = item["title"]
                        score = item.get("score")

                        movie = search_movie(title_str)
                        if not movie:
                            movie = {
                                "id": hash(title_str),
                                "title": title_str,
                                "vote_average": item.get("vote_average", "N/A"),
                                "release_date": "",
                                "overview": item.get("overview") or "No description available.",
                                "poster_path": None,
                            }

                        if not p_url and movie.get("poster_path"):
                            p_url = IMAGE_URL + movie["poster_path"]

                        col1, col2 = st.columns([1, 4])

                        with col1:
                            if p_url:
                                st.image(
                                    p_url,
                                    use_container_width=True
                                )
                            else:
                                st.markdown(
                                    f"""
                                    <div class="poster-placeholder" style="height: 160px;">
                                        <div style="font-size: 32px;">🎬</div>
                                        <div style="font-size: 11px;">Movie Explorer</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                        with col2:
                            st.subheader(
                                movie.get("title", title_str)
                            )

                            rating_str = movie.get("vote_average", "N/A")
                            if rating_str == "N/A" and item.get("vote_average") is not None:
                                rating_str = item["vote_average"]
                            rating_disp = f"{rating_str:.1f}/10" if isinstance(rating_str, (int, float)) else str(rating_str)
                            score_disp = f" | 🎯 Similarity Match: {score:.1%}" if score is not None else ""
                            st.write(f"⭐ Rating: {rating_disp}{score_disp}")

                            if movie.get("release_date"):
                                st.write(f"📅 Release: {movie['release_date']}")

                            overview_text = movie.get("overview") or item.get("overview") or "No description available."
                            st.write(f"📝 {overview_text}")

                        st.divider()

                else:
                    st.info("No recommendations found.")

            except ValueError as ve:
                st.error(str(ve))
            except Exception as exc:
                st.error(f"Recommendation error: {exc}")


# =========================================================
# CATEGORY PAGE
# =========================================================

elif page == "🎭 Categories":

    st.title("🎭 Browse by Category")

    genres = {
        "Action": 28,
        "Adventure": 12,
        "Animation": 16,
        "Comedy": 35,
        "Crime": 80,
        "Drama": 18,
        "Horror": 27,
        "Romance": 10749,
        "Science Fiction": 878,
        "Thriller": 53
    }

    selected_genre = st.selectbox(
        "Select Genre",
        list(genres.keys())
    )

    if st.button("Show Movies"):

        movies = get_movies_by_genre(
            selected_genre,
            genres[selected_genre]
        )

        st.subheader(
            f"🎬 {selected_genre} Movies"
        )

        poster_urls = fetch_posters_parallel(movies[:10])
        cols = st.columns(5)

        for i, (movie, p_url) in enumerate(zip(movies[:10], poster_urls)):

            with cols[i % 5]:

                display_movie(movie, poster_url=p_url)

                if st.button(
                    "View Details",
                    key=f"genre_{movie.get('id', i)}"
                ):

                    st.session_state["selected_movie"] = movie
                    st.rerun()


# =========================================================
# MOVIE DETAILS MODAL / DRAWER
# =========================================================

if "selected_movie" in st.session_state:

    movie = st.session_state["selected_movie"]

    st.divider()

    st.title(
        f"🎬 {movie.get('title', 'Movie Details')}"
    )

    col1, col2 = st.columns([1, 2])

    with col1:

        p_url = movie.get("poster_url")
        if not p_url and movie.get("poster_path"):
            p_url = IMAGE_URL + movie["poster_path"]
        if not p_url and movie.get("title"):
            rel = movie.get("release_date", "")
            y = rel[:4] if rel and len(rel) >= 4 else None
            p_url = get_poster_url(movie["title"], year=y)

        if p_url:

            st.image(
                p_url,
                use_container_width=True
            )
        else:
            st.markdown(
                f"""
                <div class="poster-placeholder" style="height: 320px;">
                    <div style="font-size: 60px;">🎬</div>
                    <div style="font-size: 16px; margin-top: 10px;">{movie.get('title', '')}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with col2:

        st.header(
            movie.get("title", "Unknown")
        )

        rating_val = movie.get('vote_average')
        if rating_val is not None and isinstance(rating_val, (int, float)):
            st.write(f"⭐ Rating: {rating_val:.1f}/10")
        else:
            st.write(f"⭐ Rating: {rating_val or 'N/A'}")

        if movie.get('release_date'):
            st.write(f"📅 Release Date: {movie.get('release_date')}")

        st.write(
            f"📝 Overview:\n\n{movie.get('overview', 'No overview available.')}"
        )

        if st.button("❌ Close Movie Details"):

            del st.session_state["selected_movie"]
            st.rerun()