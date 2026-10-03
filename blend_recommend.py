"""
blend_recommend.py
==================
Blended two-movie recommendation logic for MovieMate.

Reuses the already-loaded dataframe and TF-IDF matrix from main.py.

Public API
----------
blend_recommend(title_a, title_b, top_n=10)
    -> (mode_label: str, results: list[dict])

Each result dict has keys:
    title        str
    score        float   (cosine similarity to blended vector, 0-1)
    vote_average float | None
    overview     str
    poster_path  None
"""

from __future__ import annotations

import ast
import re
from typing import Optional, Set, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

import main

# Cache for parsed genres across the full DataFrame
_CACHED_PARSED_GENRES: Optional[List[Set[str]]] = None


# ---------------------------------------------------------------------------
# Genre parsing
# ---------------------------------------------------------------------------

def _parse_genres(raw) -> Set[str]:
    """
    Parse the genres field into a normalised set of genre name strings.

    Handles every format that may exist in this dataset:
    * Space-separated plain string  -> "Animation Comedy Family" or "Action Science Fiction"
    * Python list-of-dicts string   -> "[{'id':16,'name':'Animation'}, ...]"
    * Python list-of-strings string -> "['Animation', 'Comedy']"
    * Pipe or comma separated       -> "Animation|Comedy" or "Animation, Comedy"
    * Actual list/dict objects      -> already parsed by pandas
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return set()

    # Already a list / tuple / set
    if isinstance(raw, (list, tuple, set)):
        genres: Set[str] = set()
        for item in raw:
            if isinstance(item, dict):
                n = item.get("name") or item.get("genre") or ""
                if n:
                    genres.add(str(n).strip().title())
            elif isinstance(item, str) and item.strip():
                genres.add(item.strip().title())
        return genres

    raw_str = str(raw).strip()
    if not raw_str:
        return set()

    # Try ast.literal_eval for list-like strings
    if raw_str.startswith("[") and raw_str.endswith("]"):
        try:
            parsed = ast.literal_eval(raw_str)
            if isinstance(parsed, (list, tuple)):
                return _parse_genres(parsed)
        except Exception:
            pass

    # Pipe separated
    if "|" in raw_str:
        return {g.strip().title() for g in raw_str.split("|") if g.strip()}

    # Comma separated
    if "," in raw_str:
        return {g.strip().title() for g in raw_str.split(",") if g.strip()}

    # Space-separated (actual format in this dataset)
    # Preserve known multi-word genres (e.g. Science Fiction, TV Movie)
    s = raw_str
    found_multi: Set[str] = set()
    for mw in ("Science Fiction", "TV Movie"):
        if mw.lower() in s.lower():
            found_multi.add(mw)
            s = re.sub(re.escape(mw), " ", s, flags=re.IGNORECASE)

    parts = {p.strip().title() for p in s.split() if p.strip()}
    return parts | found_multi


def _get_all_parsed_genres(df: pd.DataFrame) -> List[Set[str]]:
    """Return cached list of parsed genre sets for all rows in df."""
    global _CACHED_PARSED_GENRES
    if _CACHED_PARSED_GENRES is None or len(_CACHED_PARSED_GENRES) != len(df):
        _CACHED_PARSED_GENRES = [_parse_genres(g) for g in df["genres"]]
    return _CACHED_PARSED_GENRES


# ---------------------------------------------------------------------------
# Movie lookup
# ---------------------------------------------------------------------------

def _lookup_movie(title: str) -> Optional[Tuple[int, pd.Series]]:
    """
    Return (row_index, row_series) for *title* or None if not found.
    Tries exact case-insensitive match first, then partial substring match.
    """
    main.ensure_models_loaded()
    title_to_idx = main.TITLE_TO_IDX
    df = main.df

    if title_to_idx is None or df is None:
        return None

    clean = str(title).strip()
    if not clean:
        return None

    key = clean.lower()

    # 1. Exact case-insensitive match
    if key in title_to_idx:
        idx = int(title_to_idx[key])
        return idx, df.iloc[idx]

    # 2. Partial: query is contained in stored title (e.g. "nemo" in "finding nemo")
    partial_matches = [
        (stored_key, idx) for stored_key, idx in title_to_idx.items()
        if key in stored_key
    ]
    if partial_matches:
        # Prefer exact word match or prefix match or shortest title
        partial_matches.sort(
            key=lambda x: (
                0 if x[0].startswith(key) else (1 if f" {key}" in x[0] else 2),
                len(x[0])
            )
        )
        best_idx = int(partial_matches[0][1])
        return best_idx, df.iloc[best_idx]

    return None


# ---------------------------------------------------------------------------
# Core blended recommendation
# ---------------------------------------------------------------------------

def blend_recommend(title_a: str, title_b: str, top_n: int = 10) -> Tuple[str, List[dict]]:
    """
    Return (mode_label, results).

    Raises ValueError with a descriptive message if either title is not found.
    """
    main.ensure_models_loaded()
    tfidf_matrix = main.tfidf_matrix
    df = main.df

    clean_a = str(title_a).strip()
    clean_b = str(title_b).strip()

    if not clean_a and not clean_b:
        raise ValueError("Please enter both movie names.")
    if not clean_a:
        raise ValueError("Please enter the first movie name.")
    if not clean_b:
        raise ValueError("Please enter the second movie name.")

    # Lookup both movies
    hit_a = _lookup_movie(clean_a)
    hit_b = _lookup_movie(clean_b)

    missing = []
    if hit_a is None:
        missing.append(clean_a)
    if hit_b is None:
        missing.append(clean_b)
    if missing:
        if len(missing) == 1:
            raise ValueError(f"Movie not found in dataset: '{missing[0]}'")
        else:
            raise ValueError(f"Movies not found in dataset: '{missing[0]}' and '{missing[1]}'")

    idx_a, row_a = hit_a
    idx_b, row_b = hit_b

    # Parse genres of both movies
    genres_a = _parse_genres(row_a.get("genres"))
    genres_b = _parse_genres(row_b.get("genres"))
    common_genres = genres_a & genres_b

    # Blended TF-IDF vector (average of the two sparse rows)
    vec_a = tfidf_matrix[idx_a]
    vec_b = tfidf_matrix[idx_b]
    blended_vec = (vec_a + vec_b) / 2.0

    # L2-normalize blended vector to get true cosine similarity scores
    blended_norm = sp.linalg.norm(blended_vec)
    if blended_norm > 0:
        blended_vec_norm = blended_vec / blended_norm
    else:
        blended_vec_norm = blended_vec

    scores_arr = (tfidf_matrix @ blended_vec_norm.T).toarray().ravel()

    # Candidate filtering
    exclude_indices = {idx_a, idx_b}
    all_movie_genres = _get_all_parsed_genres(df)

    if common_genres:
        # Candidates containing all shared genres
        candidate_indices = [
            i for i, g_set in enumerate(all_movie_genres)
            if i not in exclude_indices and common_genres.issubset(g_set)
        ]

        # If fewer than top_n, loosen filter to movies containing any shared genre
        if len(candidate_indices) < top_n:
            candidate_indices = [
                i for i, g_set in enumerate(all_movie_genres)
                if i not in exclude_indices and bool(common_genres & g_set)
            ]

        mode_label = f"Common genre(s): {', '.join(sorted(common_genres))}"
    else:
        candidate_indices = [i for i in range(len(df)) if i not in exclude_indices]
        mode_label = "No common genre, matched by overview similarity"

    # Rank filtered candidates by blended cosine similarity
    ranked = sorted(candidate_indices, key=lambda i: scores_arr[i], reverse=True)

    # Exclude both input movies by title as well
    seen_titles = {
        str(row_a["title"]).strip().lower(),
        str(row_b["title"]).strip().lower(),
        clean_a.lower(),
        clean_b.lower(),
    }

    results: List[dict] = []
    for i in ranked:
        row = df.iloc[i]
        t = str(row["title"])
        t_low = t.strip().lower()
        if t_low in seen_titles:
            continue
        seen_titles.add(t_low)

        rating = row.get("vote_average")
        results.append(
            {
                "title": t,
                "score": float(scores_arr[i]),
                "vote_average": float(rating) if pd.notna(rating) else None,
                "overview": str(row.get("overview", "")).strip() or "No description available.",
                "poster_path": None,
            }
        )
        if len(results) >= top_n:
            break

    return mode_label, results
