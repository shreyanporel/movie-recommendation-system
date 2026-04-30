import pandas as pd
import numpy as np
from collections import defaultdict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# LOAD DATA
# -------------------------------
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")


# -------------------------------
# FILTER DATA (SCALABILITY)
# -------------------------------
TOP_K_MOVIES = 5000

popular_ids = ratings.groupby("movieId").size() \
    .sort_values(ascending=False) \
    .head(TOP_K_MOVIES).index

filtered_movies = movies[movies["movieId"].isin(popular_ids)].reset_index(drop=True)


# -------------------------------
# BUILD MODEL
# -------------------------------
filtered_movies["genres"] = filtered_movies["genres"].fillna("")

filtered_movies["content"] = (
    filtered_movies["title"] + " " +
    filtered_movies["genres"] + " " +
    filtered_movies["genres"] + " " +
    filtered_movies["genres"]
)

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(filtered_movies["content"])


# -------------------------------
# ID → INDEX MAP
# -------------------------------
movie_id_to_index = {
    movie_id: idx for idx, movie_id in enumerate(filtered_movies["movieId"])
}


# -------------------------------
# POPULARITY
# -------------------------------
popularity_scores = ratings.groupby("movieId").size().sort_values(ascending=False)


# -------------------------------
# FORMAT OUTPUT
# -------------------------------
def format_movies(indices, scores=None):
    results = []

    for i, idx in enumerate(indices):
        movie = filtered_movies.iloc[idx]

        item = {
            "movieId": int(movie["movieId"]),
            "title": movie["title"],
            "genres": movie["genres"].split("|")
        }

        if scores is not None:
            item["score"] = round(min(scores[i] * 120, 100), 2)

        results.append(item)

    return results


# =========================================================
# PIPELINE STARTS HERE
# =========================================================

# -------------------------------
# 1. RETRIEVE (~200 candidates)
# -------------------------------
def retrieve_candidates(user_id):
    user_movies = ratings[ratings["userId"] == user_id]["movieId"].tolist()

    candidates = set()

    # ---- Content Retrieval ----
    for movie_id in user_movies[-5:]:
        if movie_id not in movie_id_to_index:
            continue

        idx = movie_id_to_index[movie_id]
        scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

        top_indices = scores.argsort()[::-1][1:21]  # top 20
        candidates.update(top_indices)

    # ---- Collaborative Retrieval ----
    similar_users = ratings[ratings["movieId"].isin(user_movies)]["userId"].unique()

    similar_data = ratings[
        (ratings["userId"].isin(similar_users)) &
        (ratings["rating"] >= 4)
    ]

    collab_movies = similar_data["movieId"].value_counts().head(100).index

    for mid in collab_movies:
        if mid in movie_id_to_index:
            candidates.add(movie_id_to_index[mid])

    # ---- Popular Retrieval ----
    popular_movies = popularity_scores.head(50).index

    for mid in popular_movies:
        if mid in movie_id_to_index:
            candidates.add(movie_id_to_index[mid])

    return list(candidates)[:200]


# -------------------------------
# 2. SCORE (only candidates)
# -------------------------------
def score_candidates(user_id, candidates):
    score_map = defaultdict(float)

    user_data = ratings[(ratings["userId"] == user_id) & (ratings["rating"] >= 4)]
    user_movies = user_data["movieId"].tolist()

    recent_movies = user_movies[-10:]
    weights = np.linspace(1, 2, len(recent_movies)) if recent_movies else []

    for movie_id, weight in zip(recent_movies, weights):
        if movie_id not in movie_id_to_index:
            continue

        idx = movie_id_to_index[movie_id]
        scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

        for c in candidates:
            score_map[c] += scores[c] * weight

    return score_map


# -------------------------------
# 3. RANK (final output)
# -------------------------------
def rank_results(score_map, user_id, top_n=10):
    watched = set(ratings[ratings["userId"] == user_id]["movieId"])

    filtered = [
        (idx, score)
        for idx, score in score_map.items()
        if filtered_movies.iloc[idx]["movieId"] not in watched
    ]

    filtered.sort(key=lambda x: x[1], reverse=True)

    indices = [idx for idx, _ in filtered[:top_n]]
    scores = [score for _, score in filtered[:top_n]]

    return format_movies(indices, scores)


# -------------------------------
# FINAL PIPELINE FUNCTION
# -------------------------------
def get_recommendations(user_id):
    candidates = retrieve_candidates(user_id)

    if not candidates:
        return get_trending_movies()

    score_map = score_candidates(user_id, candidates)

    return rank_results(score_map, user_id)


# -------------------------------
# TRENDING
# -------------------------------
def get_trending_movies(top_n=10):
    top_movies = popularity_scores.head(top_n).index.tolist()

    indices = [
        movie_id_to_index[mid]
        for mid in top_movies
        if mid in movie_id_to_index
    ]

    return format_movies(indices)


# -------------------------------
# CACHE
# -------------------------------
user_cache = {}

def get_cached_recommendations(user_id):
    if user_id not in user_cache:
        user_cache[user_id] = get_recommendations(user_id)
    return user_cache[user_id]


# -------------------------------
# FASTAPI SETUP
# -------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# API ENDPOINTS
# -------------------------------
@app.get("/")
def root():
    return {"message": "Pipeline-based Recommendation API Running"}


@app.get("/recommend/{user_id}")
def recommend_api(user_id: int):
    return {
        "user_id": user_id,
        "recommendations": get_cached_recommendations(user_id)
    }


@app.get("/similar/{movie_id}")
def similar_api(movie_id: int):
    if movie_id not in movie_id_to_index:
        return {"error": "Movie not found"}

    idx = movie_id_to_index[movie_id]
    scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    top_indices = scores.argsort()[::-1][1:11]

    return {
        "based_on": movie_id,
        "similar_movies": format_movies(top_indices)
    }


@app.get("/search")
def search_movies(query: str):
    query_vec = vectorizer.transform([query.lower()])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = [i for i in scores.argsort()[::-1] if scores[i] > 0.2][:10]

    return {"results": format_movies(top_indices, scores[top_indices])}


@app.get("/trending")
def trending_api():
    return {"trending": get_trending_movies()}