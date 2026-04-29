import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI


# -------------------------------
# Load Data
# -------------------------------
def load_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    return movies, ratings


# -------------------------------
# Build Model
# -------------------------------
def build_model(movies):
    movies["genres"] = movies["genres"].fillna("")

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(movies["genres"])

    return tfidf_matrix


# -------------------------------
# Format Output for Frontend
# -------------------------------
def format_movies(indices, movies, scores=None):
    results = []

    for i, idx in enumerate(indices):
        movie = movies.iloc[idx]
        item = {
            "movieId": int(movie["movieId"]),
            "title": movie["title"],
            "genres": movie["genres"].split("|")
        }

        if scores is not None:
            item["score"] = round(float(scores[i]* 100),2)  # Scale scores to 0-100

        results.append(item)

    return results


# -------------------------------
# Recommend Similar Movies
# -------------------------------
def recommend(movie_id, movies, tfidf_matrix, top_n=5):
    if movie_id not in movies["movieId"].values:
        return []

    idx = movies[movies["movieId"] == movie_id].index[0]

    scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    sorted_indices = scores.argsort()[::-1]

    sorted_indices = [i for i in sorted_indices if i != idx]
    top_indices = sorted_indices[:top_n]
    top_scores = scores[top_indices]
    return format_movies(top_indices, movies, top_scores)


# -------------------------------
# Trending Movies
# -------------------------------
def get_trending_movies(ratings, movies, top_n=5):
    popular = ratings.groupby("movieId").size().sort_values(ascending=False).head(top_n)
    indices = movies[movies["movieId"].isin(popular.index)].index.tolist()

    return format_movies(indices, movies)


# -------------------------------
# User-Based Recommendation
# -------------------------------
def get_user_recommendations(user_id, movies, ratings, tfidf_matrix):
    user_data = ratings[(ratings["userId"] == user_id) & (ratings["rating"] >= 4)]
    user_movies = user_data["movieId"].tolist()

    # Cold start
    if not user_movies:
        return get_trending_movies(ratings, movies)

    recommendations = []
    scores = []

    for movie in user_movies[-3:]:
        idx = movies[movies["movieId"] == movie].index[0]

        sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        top_indices = sim_scores.argsort()[::-1][1:11]

        for i in top_indices:
            recommendations.append(i)
            scores.append(sim_scores[i])

    watched_indices = movies[movies["movieId"].isin(user_movies)].index.tolist()

    final = []
    final_scores = []

    for i, idx in enumerate(recommendations):
        if idx not in watched_indices and idx not in final:
            final.append(idx)
            final_scores.append(scores[i])

    return format_movies(final[:5], movies, final_scores[:5])


# -------------------------------
# FastAPI Setup
# -------------------------------
app = FastAPI()

movies, ratings = load_data()
tfidf_matrix = build_model(movies)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# API Endpoints
# -------------------------------

@app.get("/")
def home_root():
    return {"message": "Recommendation API is running"}


@app.get("/recommend/{user_id}")
def recommend_api(user_id: int):
    return {
        "user_id": user_id,
        "recommendations": get_user_recommendations(user_id, movies, ratings, tfidf_matrix)
    }


@app.get("/similar/{movie_id}")
def similar_api(movie_id: int):
    movie_row = movies[movies["movieId"] == movie_id]

    if movie_row.empty:
        return {"error": "Movie not found"}

    movie_title = movie_row.iloc[0]["title"]

    return {
        "movie_id": movie_id,
        "based_on": movie_title,
        "similar_movies": recommend(movie_id, movies, tfidf_matrix)
    }   


@app.get("/trending")
def trending_api():
    return {
        "trending": get_trending_movies(ratings, movies)
    }


@app.get("/home/{user_id}")
def home(user_id: int):
    return {
        "user_id": user_id,
        "for_you": get_user_recommendations(user_id, movies, ratings, tfidf_matrix),
        "trending": get_trending_movies(ratings, movies)
    }

@app.get("/search")
def search_movies(query: str):
    query = query.lower()

    results = movies[movies["title"].str.lower().str.contains(query, na=False)]

    # Limit results
    results = results.head(10)

    formatted = [
        {
            "movieId": int(row["movieId"]),
            "title": row["title"],
            "genres": row["genres"].split("|")
        }
        for _, row in results.iterrows()
    ]

    return {"results": formatted}
# -------------------------------
# Local Testing
# -------------------------------
if __name__ == "__main__":
    print("Testing system...\n")

    print("For User 1:")
    print(get_user_recommendations(1, movies, ratings, tfidf_matrix))

    print("\nCold Start (New User):")
    print(get_user_recommendations(999999, movies, ratings, tfidf_matrix))
    
    

    