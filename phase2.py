"""
CineMatch Phase 2 — Netflix-style Recommendation Engine
========================================================
Architecture:
  1. FAISS ANN Index        → ultra-fast approximate nearest-neighbour retrieval
  2. Retrieve → Score → Rank pipeline (kept from Phase 1, upgraded)
  3. Redis-style in-process TTL cache  → hot user profiles served in <1 ms
  4. Event-driven update bus           → rating events invalidate stale cache entries
  5. Background worker (asyncio)       → rebuilds affected user recs asynchronously
"""

import asyncio
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# FAISS — install via:  pip install faiss-cpu
import faiss

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TOP_K_MOVIES         = 5_000   # movies to keep in index
FAISS_NLIST          = 64      # IVF clusters (tune ↑ for bigger corpora)
FAISS_NPROBE         = 8       # clusters searched at query time (speed ↔ recall)
RETRIEVE_PER_SEED    = 30      # FAISS neighbours per seed movie
RETRIEVE_MAX         = 300     # hard cap on candidate set
COLLAB_TOP_USERS     = 200     # similar users to consider
COLLAB_TOP_MOVIES    = 100     # top movies from similar users
POPULAR_SEED         = 50      # always include N popular movies as candidates
SCORE_RECENT_WINDOW  = 10      # last N rated movies used for scoring
SCORE_WEIGHT_RANGE   = (1.0, 2.5)  # linear ramp — older→newer
RANK_TOP_N           = 10      # final recommendations returned

CACHE_TTL_SECONDS    = 300     # 5-minute TTL on cached recs
CACHE_MAX_ENTRIES    = 10_000  # evict LRU when exceeded


# ─────────────────────────────────────────────────────────────────────────────
# REDIS-STYLE TTL CACHE  (in-process, LRU + TTL)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _CacheEntry:
    value: Any
    expires_at: float
    last_used: float = field(default_factory=time.monotonic)


class TTLCache:
    """
    Thread-safe (GIL-protected) dictionary with per-entry TTL and LRU eviction.
    Mirrors the interface of a Redis client without network overhead for
    single-process deployments. Drop-in replace with redis-py for multi-process.
    """

    def __init__(self, ttl: float = CACHE_TTL_SECONDS, maxsize: int = CACHE_MAX_ENTRIES):
        self._store: Dict[str, _CacheEntry] = {}
        self._ttl   = ttl
        self._max   = maxsize

    # ---- public API ----

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.monotonic() > entry.expires_at:
            del self._store[key]
            return None
        entry.last_used = time.monotonic()
        return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        if len(self._store) >= self._max:
            self._evict_lru()
        self._store[key] = _CacheEntry(
            value      = value,
            expires_at = time.monotonic() + (ttl or self._ttl),
        )

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def invalidate_prefix(self, prefix: str) -> int:
        """Bulk-delete all keys starting with prefix. Returns count deleted."""
        victims = [k for k in self._store if k.startswith(prefix)]
        for k in victims:
            del self._store[k]
        return len(victims)

    def stats(self) -> dict:
        now = time.monotonic()
        live = sum(1 for e in self._store.values() if e.expires_at > now)
        return {"total_keys": len(self._store), "live_keys": live, "ttl": self._ttl}

    # ---- internals ----

    def _evict_lru(self) -> None:
        if not self._store:
            return
        lru_key = min(self._store, key=lambda k: self._store[k].last_used)
        del self._store[lru_key]


# Singleton cache instances — one per logical namespace (mirrors Redis keyspaces)
rec_cache      = TTLCache(ttl=300)   # user recommendations
similar_cache  = TTLCache(ttl=600)   # item-item similarity
search_cache   = TTLCache(ttl=120)   # full-text search results
trending_cache = TTLCache(ttl=60)    # trending list (changes fast)


# ─────────────────────────────────────────────────────────────────────────────
# EVENT BUS  (lightweight pub/sub, swap for Kafka/SQS in production)
# ─────────────────────────────────────────────────────────────────────────────

EventHandler = Callable[..., Any]


class EventBus:
    """
    In-process async pub/sub.
    Producers call  bus.emit(event, payload)
    Consumers call  bus.subscribe(event, handler)
    Handlers run as asyncio tasks — fire and forget, non-blocking.
    """

    def __init__(self):
        self._handlers: Dict[str, List[EventHandler]] = defaultdict(list)

    def subscribe(self, event: str, handler: EventHandler) -> None:
        self._handlers[event].append(handler)

    async def emit(self, event: str, payload: dict) -> None:
        for handler in self._handlers.get(event, []):
            asyncio.create_task(self._safe_call(handler, payload))

    @staticmethod
    async def _safe_call(fn: EventHandler, payload: dict) -> None:
        try:
            result = fn(payload)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:                       # noqa: BLE001
            print(f"[EventBus] handler error: {exc}")


bus = EventBus()


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

print("[boot] Loading datasets…")
movies  = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

# Filter to top-K most-rated movies for scalability
popular_ids = (
    ratings.groupby("movieId").size()
    .sort_values(ascending=False)
    .head(TOP_K_MOVIES).index
)
fmovies = movies[movies["movieId"].isin(popular_ids)].reset_index(drop=True)

# Popularity lookup (used throughout)
popularity_scores = ratings.groupby("movieId").size().sort_values(ascending=False)

# movieId  ←→  positional index in fmovies
id_to_idx: Dict[int, int] = {mid: i for i, mid in enumerate(fmovies["movieId"])}


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF VECTORISATION
# ─────────────────────────────────────────────────────────────────────────────

print("[boot] Building TF-IDF matrix…")
fmovies["genres"] = fmovies["genres"].fillna("")

# Repeat genre tokens 3× to give them more weight than title tokens
fmovies["content"] = (
    fmovies["title"] + " "
    + fmovies["genres"] + " "
    + fmovies["genres"] + " "
    + fmovies["genres"]
)

vectorizer  = TfidfVectorizer(stop_words="english")
tfidf_csr   = vectorizer.fit_transform(fmovies["content"])          # sparse CSR
tfidf_dense = np.array(tfidf_csr.todense(), dtype=np.float32)       # dense F32 for FAISS


# ─────────────────────────────────────────────────────────────────────────────
# FAISS INDEX  (IVF + flat L2, equivalent to cosine after L2-normalisation)
# ─────────────────────────────────────────────────────────────────────────────

print("[boot] Building FAISS index…")

# L2-normalise so inner-product ≡ cosine similarity
faiss.normalize_L2(tfidf_dense)

dim       = tfidf_dense.shape[1]
quantiser = faiss.IndexFlatIP(dim)                               # exact inner-product base
index     = faiss.IndexIVFFlat(quantiser, dim, FAISS_NLIST, faiss.METRIC_INNER_PRODUCT)

index.train(tfidf_dense)
index.add(tfidf_dense)
index.nprobe = FAISS_NPROBE                                      # search NPROBE clusters

print(f"[boot] FAISS index ready — {index.ntotal:,} vectors, dim={dim}")


def faiss_knn(query_idx: int, k: int = RETRIEVE_PER_SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Return (indices, scores) for the k nearest neighbours of query_idx."""
    vec = tfidf_dense[query_idx : query_idx + 1]                 # already normalised
    scores, indices = index.search(vec, k + 1)                   # +1 to exclude self
    # Strip self (score ≈ 1.0)
    mask = indices[0] != query_idx
    return indices[0][mask][:k], scores[0][mask][:k]


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEP 1 — RETRIEVE  (~300 candidates)
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_candidates(user_id: int) -> List[int]:
    """
    Three-source candidate generation:
      A) Content   — FAISS ANN on recently-rated seed movies
      B) Collab    — highly-rated movies from similar users
      C) Popularity — always include popular titles as a safety net
    """
    user_movies = ratings[ratings["userId"] == user_id]["movieId"].tolist()

    candidates: set[int] = set()

    # ── A. Content-based via FAISS ──────────────────────────────────────────
    for movie_id in user_movies[-5:]:
        if movie_id not in id_to_idx:
            continue
        nn_indices, _ = faiss_knn(id_to_idx[movie_id], k=RETRIEVE_PER_SEED)
        candidates.update(nn_indices.tolist())

    # ── B. Collaborative filtering ──────────────────────────────────────────
    similar_users = (
        ratings[ratings["movieId"].isin(user_movies)]["userId"]
        .value_counts().head(COLLAB_TOP_USERS).index
    )
    collab_movies = (
        ratings[
            ratings["userId"].isin(similar_users) & (ratings["rating"] >= 4)
        ]["movieId"].value_counts().head(COLLAB_TOP_MOVIES).index
    )
    for mid in collab_movies:
        if mid in id_to_idx:
            candidates.add(id_to_idx[mid])

    # ── C. Popularity safety net ─────────────────────────────────────────────
    for mid in popularity_scores.head(POPULAR_SEED).index:
        if mid in id_to_idx:
            candidates.add(id_to_idx[mid])

    return list(candidates)[:RETRIEVE_MAX]


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEP 2 — SCORE  (recency-weighted cosine)
# ─────────────────────────────────────────────────────────────────────────────

def score_candidates(user_id: int, candidates: List[int]) -> Dict[int, float]:
    """
    For each recently-liked movie, compute cosine similarity to all candidates
    using the dense FAISS vectors (fast) and aggregate with a recency weight
    (newer ratings matter more).
    """
    user_data   = ratings[(ratings["userId"] == user_id) & (ratings["rating"] >= 4)]
    user_movies = user_data["movieId"].tolist()
    seeds       = user_movies[-SCORE_RECENT_WINDOW:]

    if not seeds:
        return {}

    weights = np.linspace(*SCORE_WEIGHT_RANGE, len(seeds))
    score_map: Dict[int, float] = defaultdict(float)

    cand_arr = np.array(candidates, dtype=np.int64)
    cand_vecs = tfidf_dense[cand_arr]                            # (C, D)

    for seed_movie_id, w in zip(seeds, weights):
        if seed_movie_id not in id_to_idx:
            continue
        seed_vec = tfidf_dense[id_to_idx[seed_movie_id]]        # (D,)
        sims     = cand_vecs @ seed_vec                          # (C,) dot product = cosine (normalised)
        for cand_idx, sim in zip(candidates, sims):
            score_map[cand_idx] += float(sim) * w

    return score_map


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEP 3 — RANK  (filter watched, sort, return top-N)
# ─────────────────────────────────────────────────────────────────────────────

def rank_results(score_map: Dict[int, float], user_id: int, top_n: int = RANK_TOP_N) -> List[dict]:
    watched_ids = set(ratings[ratings["userId"] == user_id]["movieId"])

    ranked = [
        (idx, score)
        for idx, score in score_map.items()
        if fmovies.iloc[idx]["movieId"] not in watched_ids
    ]
    ranked.sort(key=lambda x: x[1], reverse=True)

    indices = [i for i, _ in ranked[:top_n]]
    scores  = [s for _, s in ranked[:top_n]]
    return _format(indices, scores)


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FORMATTER
# ─────────────────────────────────────────────────────────────────────────────

def _format(indices: List[int], scores: Optional[List[float]] = None) -> List[dict]:
    results = []
    for i, idx in enumerate(indices):
        row  = fmovies.iloc[idx]
        item = {
            "movieId": int(row["movieId"]),
            "title"  : row["title"],
            "genres" : row["genres"].split("|"),
        }
        if scores is not None:
            # Clamp to 0-100 display scale
            item["score"] = round(min(float(scores[i]) * 120, 100.0), 2)
        results.append(item)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL RECOMMENDATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline(user_id: int) -> List[dict]:
    candidates = retrieve_candidates(user_id)
    if not candidates:
        return _get_trending()
    score_map = score_candidates(user_id, candidates)
    return rank_results(score_map, user_id)


def get_recommendations(user_id: int) -> List[dict]:
    """Cache-first recommendation fetch."""
    key    = f"rec:{user_id}"
    cached = rec_cache.get(key)
    if cached is not None:
        return cached
    result = _run_pipeline(user_id)
    rec_cache.set(key, result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# TRENDING
# ─────────────────────────────────────────────────────────────────────────────

def _get_trending(top_n: int = 10) -> List[dict]:
    cached = trending_cache.get("trending")
    if cached:
        return cached
    top_ids = popularity_scores.head(top_n).index.tolist()
    indices = [id_to_idx[mid] for mid in top_ids if mid in id_to_idx]
    result  = _format(indices)
    trending_cache.set("trending", result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# EVENT HANDLERS  (subscribed to the bus)
# ─────────────────────────────────────────────────────────────────────────────

async def on_rating_submitted(payload: dict) -> None:
    """
    Fired when a user submits a rating.
    1. Append to in-memory ratings DataFrame (hot path — no disk write needed
       for recommendation freshness; persist separately via your DB layer).
    2. Invalidate the user's rec cache entry.
    3. Schedule an async rebuild so the next request hits a warm cache.
    """
    global ratings

    user_id  = payload["userId"]
    movie_id = payload["movieId"]
    rating   = payload["rating"]

    # 1. Append to live ratings
    new_row = pd.DataFrame([{"userId": user_id, "movieId": movie_id, "rating": rating}])
    ratings = pd.concat([ratings, new_row], ignore_index=True)

    # 2. Invalidate stale cache
    n = rec_cache.invalidate_prefix(f"rec:{user_id}")
    print(f"[event] rating_submitted uid={user_id} mid={movie_id} → invalidated {n} cache entries")

    # 3. Async rebuild (runs in background, doesn't block API response)
    await _rebuild_user_cache(user_id)


async def on_movie_added(payload: dict) -> None:
    """
    Fired when a new movie is added to the catalogue.
    Invalidate all similarity caches (they reference a now-stale FAISS index).
    A full index rebuild would be triggered here in production.
    """
    n = similar_cache.invalidate_prefix("sim:")
    print(f"[event] movie_added mid={payload.get('movieId')} → invalidated {n} sim-cache entries")


async def on_popularity_shift(payload: dict) -> None:
    """Fired by a scheduled job when trending list changes significantly."""
    trending_cache.delete("trending")
    print("[event] popularity_shift → trending cache cleared")


# Register all handlers
bus.subscribe("rating_submitted",  on_rating_submitted)
bus.subscribe("movie_added",       on_movie_added)
bus.subscribe("popularity_shift",  on_popularity_shift)


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND WORKER — async cache rebuilder
# ─────────────────────────────────────────────────────────────────────────────

async def _rebuild_user_cache(user_id: int) -> None:
    """
    Offloaded to a background asyncio task.
    Runs the full pipeline and populates the cache so the next HTTP request
    is served instantly.
    """
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run_pipeline, user_id)
    rec_cache.set(f"rec:{user_id}", result)
    print(f"[worker] cache rebuilt for user {user_id}")


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="CineMatch Phase 2", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic request models ──────────────────────────────────────────────────

class RatingEvent(BaseModel):
    userId : int
    movieId: int
    rating : float           # 0.5 – 5.0


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "CineMatch Phase 2 running", "index_size": index.ntotal}


@app.get("/recommend/{user_id}")
def recommend_api(user_id: int):
    """
    Returns personalised recommendations.
    Cache hit  → <1 ms  (TTL cache)
    Cache miss → runs Retrieve→Score→Rank pipeline
    """
    t0   = time.perf_counter()
    recs = get_recommendations(user_id)
    ms   = round((time.perf_counter() - t0) * 1000, 2)
    return {"user_id": user_id, "recommendations": recs, "latency_ms": ms}


@app.get("/similar/{movie_id}")
def similar_api(movie_id: int):
    """
    Item-item similarity via FAISS ANN.
    Cached per movie_id with 10-minute TTL.
    """
    key    = f"sim:{movie_id}"
    cached = similar_cache.get(key)
    if cached:
        return cached

    if movie_id not in id_to_idx:
        raise HTTPException(status_code=404, detail="Movie not found")

    nn_idx, nn_scores = faiss_knn(id_to_idx[movie_id], k=10)
    result = {
        "based_on"     : movie_id,
        "similar_movies": _format(nn_idx.tolist(), nn_scores.tolist()),
    }
    similar_cache.set(key, result)
    return result


@app.get("/search")
def search_api(q: str = Query(..., min_length=2)):
    """
    Full-text semantic search using TF-IDF + cosine.
    Results cached for 2 minutes per query hash.
    """
    cache_key = "search:" + hashlib.md5(q.lower().encode()).hexdigest()
    cached    = search_cache.get(cache_key)
    if cached:
        return cached

    qvec   = vectorizer.transform([q.lower()])
    qvec_d = np.array(qvec.todense(), dtype=np.float32)
    faiss.normalize_L2(qvec_d)

    scores, indices = index.search(qvec_d, 20)
    top_idx    = [int(i) for i, s in zip(indices[0], scores[0]) if i >= 0 and s > 0.1]
    top_scores = [float(s) for i, s in zip(indices[0], scores[0]) if i >= 0 and s > 0.1]

    result = {"query": q, "results": _format(top_idx, top_scores)}
    search_cache.set(cache_key, result)
    return result


@app.get("/trending")
def trending_api():
    return {"trending": _get_trending()}


@app.post("/rate")
async def rate_movie(event: RatingEvent, background_tasks: BackgroundTasks):
    """
    Submit a rating.
    Immediately returns 202 Accepted, then asynchronously:
      - appends to ratings DataFrame
      - invalidates stale cache
      - rebuilds user recommendations in background
    """
    await bus.emit("rating_submitted", event.model_dump())
    return {"status": "accepted", "message": "Rating recorded. Recommendations updating…"}


@app.get("/cache/stats")
def cache_stats():
    """Operational view of all cache namespaces (like Redis INFO)."""
    return {
        "recommendations" : rec_cache.stats(),
        "similarity"      : similar_cache.stats(),
        "search"          : search_cache.stats(),
        "trending"        : trending_cache.stats(),
    }


@app.delete("/cache/user/{user_id}")
def bust_user_cache(user_id: int):
    """Force-invalidate a single user's recommendation cache."""
    n = rec_cache.invalidate_prefix(f"rec:{user_id}")
    return {"invalidated": n}


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL TEST HARNESS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n── Recommend user 1 (cold cache) ──")
    t = time.perf_counter()
    r = get_recommendations(1)
    print(f"  {len(r)} results in {(time.perf_counter()-t)*1000:.1f} ms")
    for m in r[:3]:
        print(f"  [{m['score']}] {m['title']}")

    print("\n── Recommend user 1 (warm cache) ──")
    t = time.perf_counter()
    get_recommendations(1)
    print(f"  served in {(time.perf_counter()-t)*1000:.3f} ms")

    print("\n── FAISS similar to movie 1 ──")
    idx_arr, sc_arr = faiss_knn(0, k=5)
    for i, s in zip(idx_arr, sc_arr):
        print(f"  [{s:.3f}] {fmovies.iloc[i]['title']}")