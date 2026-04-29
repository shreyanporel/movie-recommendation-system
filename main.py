from fastapi import FastAPI
from phase1 import load_data, build_model, get_user_recommendations

app = FastAPI()

movies, ratings = load_data()
tfidf_matrix = build_model(movies)

@app.get("/recommend/{user_id}")
def recommend_api(user_id: int):
    return {
        "user_id": user_id,
        "recommendations": get_user_recommendations(user_id, movies, ratings, tfidf_matrix)
    }

