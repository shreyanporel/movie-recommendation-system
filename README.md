# 🎬 Movie Recommendation System

A full-stack movie recommendation system using:
- FastAPI (Backend)
- TF-IDF based ML model
- HTML/CSS/JS Frontend

## Features
- Personalized recommendations
- Trending movies
- Search by movie name
- Similar movie suggestions

## How to Run
### Open folder 'Phase 1':
### Backend
uvicorn phase1:app --reload

### Frontend
Open prefront.html in browser

# 🚀 Phase 2: Scalable Recommendation System
### An upgraded version of the system with:
- FastAPI (Backend)
- FAISS (Fast similarity search)
- Retrieve–Score–Rank pipeline
- Caching (TTL + LRU)
- Async/event-driven updates
- Features
- Hybrid recommendations (content + collaborative + popularity)
- Ultra-fast recommendations (cached responses)
- Real-time updates on user activity
- Semantic search with caching
- Similar movie suggestions using FAISS
- Trending movies

## How to Run
### Open folder 'Phase 2':
### Backend
uvicorn phase2:app --reload

### Frontend
Open front.html in browser

## Dataset

Download MovieLens dataset from:
https://grouplens.org/datasets/movielens/

Place files inside:
data/

Required files:
- movies.csv
- ratings.csv
