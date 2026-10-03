import os
import pickle
from typing import Optional, List, Dict, Any, Tuple
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    print("WARNING: TMDB_API_KEY is missing in .env. TMDB features will return a 503 error until configured.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
INDICES = INDICES_PATH
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None

class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []

class TFIDFRecItem(BaseModel):
    title: str 
    score: float 
    tmdb: Optional[TMDBMovieCard] = None

class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]

def _norm_title(t: str) -> str:
    return str(t).strip().lower()

def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"

async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not TMDB_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="TMDB_API_KEY missing. Please configure TMDB_API_KEY in your .env file to enable TMDB endpoints.",
        )
    q = dict(params)
    q["api_key"] = TMDB_API_KEY
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{TMDB_BASE_URL}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB request error: {type(e).__name__} | {repr(e)}",
        )

    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB error {r.status_code}: {r.text}",
        )
    return r.json()

async def tmdb_cards_from_results(
    results: List[dict],
    limit: int = 20,
) -> List[TMDBMovieCard]:
    out: List[TMDBMovieCard] = []
    for m in (results or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or m.get("name") or "", 
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    return out

async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(
        f"/movie/{movie_id}", 
        {"language": "en-US"},
    )
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        poster_url=make_img_url(data.get("poster_path")),
        release_date=data.get("release_date"),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )

tmdb_details_from_id = tmdb_movie_details

async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    return await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
        },
    )

async def tmdb_search_first(query: str) -> Optional[dict]: 
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None

def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    title_to_idx: Dict[str, int] = {}
    try:
        for k, v in indices.items():
            norm_k = _norm_title(k)
            if norm_k not in title_to_idx:
                if hasattr(v, '__iter__') and not isinstance(v, int):
                    val = int(v.iloc[0] if hasattr(v, 'iloc') else v[0])
                else:
                    val = int(v)
                title_to_idx[norm_k] = val
        return title_to_idx
    except Exception as e:
        raise RuntimeError(
            f"indices.pkl must be dict or pandas Series-like: {e}"
        )

def ensure_models_loaded():
    global df, tfidf_matrix, TITLE_TO_IDX
    if df is None or tfidf_matrix is None or TITLE_TO_IDX is None:
        load_pickles()

def get_df() -> pd.DataFrame:
    ensure_models_loaded()
    return df

def get_title_to_idx() -> Dict[str, int]:
    ensure_models_loaded()
    return TITLE_TO_IDX


def get_local_idx_by_title(title: str) -> int:
    ensure_models_loaded()
    global TITLE_TO_IDX
    key = _norm_title(title)
    if TITLE_TO_IDX and key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    
    # Fallback substring match (query in title or title in query)
    if TITLE_TO_IDX:
        for k, idx in TITLE_TO_IDX.items():
            if key in k or (len(key) >= 3 and len(k) >= 3 and k in key):
                return int(idx)
            
    raise HTTPException(status_code=404, detail=f"Title not found in local dataset: '{title}'")

def tfidf_recommend_titles(query_title: str, top_n: int = 10) -> List[Tuple[str, float]]:
    ensure_models_loaded()
    global df, tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")
    idx = get_local_idx_by_title(query_title)
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()
    order = np.argsort(-scores)
    out: List[Tuple[str, float]] = []
    seen = set()
    norm_query = _norm_title(query_title)
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"]) 
        except Exception:
            continue
        norm_t = _norm_title(title_i)
        if norm_t == norm_query or norm_t in seen:
            continue
        seen.add(norm_t)
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out

def local_popular_cards(limit: int = 24) -> List[TMDBMovieCard]:
    ensure_models_loaded()
    if df is None:
        return []
    pop = pd.to_numeric(df.get("popularity"), errors="coerce").fillna(0)
    top_indices = pop.sort_values(ascending=False).index[:limit]
    cards: List[TMDBMovieCard] = []
    for idx in top_indices:
        row = df.iloc[idx]
        cards.append(
            TMDBMovieCard(
                tmdb_id=int(idx),
                title=str(row["title"]),
                poster_url=None,
                release_date=None,
                vote_average=float(row["vote_average"]) if pd.notna(row.get("vote_average")) else None,
            )
        )
    return cards


async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    try:
        m = await tmdb_search_first(title)
        if not m:
            return None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except Exception:
        return None

def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX
    print("Loading models and datasets...")
    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)
    
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    if df is None or "title" not in df.columns:
        raise RuntimeError("DataFrame must contain 'title' column")
    print(f"Loaded {len(TITLE_TO_IDX)} movies into index successfully.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_models_loaded()
    yield

app = FastAPI(title="Movie Recommender API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],    
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Movie Recommender API & App</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
                color: #f8fafc;
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
                margin: 0;
            }
            .card {
                background: rgba(30, 41, 59, 0.85);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
                padding: 40px;
                max-width: 520px;
                text-align: center;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
                backdrop-filter: blur(10px);
            }
            h1 { font-size: 2rem; margin-bottom: 0.5rem; color: #38bdf8; }
            p { color: #94a3b8; font-size: 1rem; line-height: 1.6; margin-bottom: 2rem; }
            .btn-group { display: flex; flex-direction: column; gap: 12px; }
            a.btn {
                display: block;
                padding: 14px 20px;
                border-radius: 10px;
                text-decoration: none;
                font-weight: 600;
                font-size: 1rem;
                transition: transform 0.15s ease, background 0.2s ease;
            }
            a.btn-primary {
                background: linear-gradient(90deg, #6366f1, #8b5cf6);
                color: white;
            }
            a.btn-primary:hover {
                background: linear-gradient(90deg, #4f46e5, #7c3aed);
                transform: translateY(-2px);
            }
            a.btn-secondary {
                background: #334155;
                color: #e2e8f0;
            }
            a.btn-secondary:hover {
                background: #475569;
                transform: translateY(-2px);
            }
            .status-badge {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 9999px;
                background: rgba(34, 197, 94, 0.15);
                color: #4ade80;
                font-size: 0.85rem;
                margin-bottom: 1rem;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <span class="status-badge">● Systems Operational</span>
            <h1>🎬 Movie Explorer</h1>
            <p>Welcome! The Movie Recommendation Engine and API are running. Choose where you want to go:</p>
            <div class="btn-group">
                <a href="http://localhost:8501" target="_blank" class="btn btn-primary">
                    👉 Open MovieMate Web App (Port 8501)
                </a>
                <a href="/docs" class="btn btn-secondary">
                    📖 Open Swagger API Docs (/docs)
                </a>
                <a href="/health" class="btn btn-secondary">
                    🩺 View Health & Metrics JSON (/health)
                </a>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/health")
def health():
    ensure_models_loaded()
    return {
        "status": "ok",
        "movies_indexed": len(TITLE_TO_IDX) if TITLE_TO_IDX else 0,
        "tmdb_configured": bool(TMDB_API_KEY)
    }

@app.get('/home', response_model=List[TMDBMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):
    if not TMDB_API_KEY:
        return local_popular_cards(limit=limit)
    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language": "en-US"}) 
            return await tmdb_cards_from_results(data.get("results", []), limit=limit)
        if category not in {"popular", "top_rated", "upcoming", "now_playing"}: 
            raise HTTPException(status_code=400, detail="Invalid category")
        data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
        return await tmdb_cards_from_results(data.get("results", []), limit=limit)
    except Exception:
        return local_popular_cards(limit=limit)

@app.get('/tmdb/search')
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10),
):
    return await tmdb_search_movies(query=query, page=page)

@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    return await tmdb_movie_details(tmdb_id)

# GENRE RECOMMENDATION
@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_genre(
    tmdb_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50),
):
    if TMDB_API_KEY:
        try:
            details = await tmdb_movie_details(tmdb_id)
            if details.genres:
                genre_id = details.genres[0]["id"]
                discover = await tmdb_get(
                    "/discover/movie",
                    {
                        "with_genres": genre_id,
                        "language": "en-US",
                        "sort_by": "popularity.desc",
                        "page": 1,
                    },
                )
                cards = await tmdb_cards_from_results(discover.get("results", []), limit=limit)
                return [c for c in cards if c.tmdb_id != tmdb_id]
        except Exception:
            pass

    # Fallback using local dataset
    ensure_models_loaded()
    cards: List[TMDBMovieCard] = []
    if df is not None and 0 <= tmdb_id < len(df):
        row = df.iloc[tmdb_id]
        genres_str = str(row.get("genres", ""))
        first_genre = genres_str.split()[0] if genres_str else ""
        if first_genre:
            matching = df[df["genres"].astype(str).str.contains(first_genre, case=False, na=False)]
            for m_idx, m_row in matching.head(limit + 1).iterrows():
                if int(m_idx) != tmdb_id:
                    cards.append(
                        TMDBMovieCard(
                            tmdb_id=int(m_idx),
                            title=str(m_row["title"]),
                            poster_url=None,
                            release_date=None,
                            vote_average=float(m_row["vote_average"]) if pd.notna(m_row.get("vote_average")) else None,
                        )
                    )
                if len(cards) >= limit:
                    break
    return cards

@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]

@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    best = None
    if TMDB_API_KEY:
        try:
            best = await tmdb_search_first(query)
        except Exception:
            best = None
    
    if best:
        tmdb_id = int(best["id"])
        details = await tmdb_movie_details(tmdb_id)
    else:
        # Fallback to local dataset
        ensure_models_loaded()
        try:
            idx = get_local_idx_by_title(query)
            row = df.iloc[idx]
            genre_list = [{"id": 0, "name": g} for g in str(row.get("genres", "")).split()]
            details = TMDBMovieDetails(
                tmdb_id=int(idx),
                title=str(row["title"]),
                overview=str(row.get("overview", "")),
                poster_url=None,
                release_date=None,
                backdrop_url=None,
                genres=genre_list,
            )
        except Exception:
            raise HTTPException(
                status_code=404, detail=f"Movie not found in TMDB or local dataset for query: {query}"
            )

    # TF-IDF recommendations (never crash endpoint)
    tfidf_items: List[TFIDFRecItem] = []
    recs: List[Tuple[str, float]] = []
    try:
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        try: 
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception: 
            recs = []
    
    for title, score in recs:
        card = await attach_tmdb_card_by_title(title) if TMDB_API_KEY else None
        tfidf_items.append(TFIDFRecItem(title=title, score=score, tmdb=card))

    # 2) Genre recommendations (TMDB discover or local fallback)
    genre_recs: List[TMDBMovieCard] = []
    if TMDB_API_KEY and details.genres and details.genres[0].get("id", 0) != 0:
        try:
            genre_id = details.genres[0]["id"]
            discover = await tmdb_get(
                "/discover/movie",
                {
                    "with_genres": genre_id,
                    "language": "en-US",
                    "sort_by": "popularity.desc",
                    "page": 1,
                },
            )
            cards = await tmdb_cards_from_results(
                discover.get("results", []), limit=genre_limit
            )
            genre_recs = [c for c in cards if c.tmdb_id != details.tmdb_id]
        except Exception:
            pass

    if not genre_recs and details.genres and df is not None:
        first_genre = details.genres[0].get("name", "") if isinstance(details.genres[0], dict) else ""
        if first_genre:
            matching = df[df["genres"].astype(str).str.contains(first_genre, case=False, na=False)]
            for m_idx, m_row in matching.head(genre_limit + 1).iterrows():
                if str(m_row["title"]).lower() != details.title.lower():
                    genre_recs.append(
                        TMDBMovieCard(
                            tmdb_id=int(m_idx),
                            title=str(m_row["title"]),
                            poster_url=None,
                            release_date=None,
                            vote_average=float(m_row["vote_average"]) if pd.notna(m_row.get("vote_average")) else None,
                        )
                    )
                if len(genre_recs) >= genre_limit:
                    break

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)