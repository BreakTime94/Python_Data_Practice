# main.py
import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from train.FM_recommend import MatrixFactorization, device

MODEL_DIR = os.path.join(os.path.dirname(__file__),"model")
MODEL_PATH = os.path.join(MODEL_DIR, "fm_model.pt")

app = FastAPI(title="MF Recommender API")

# 전역 변수
model = None
num_users = None
num_movies = None
embedding_dim = None


class PredictRequest(BaseModel):
    user_idx: int
    movie_idx: int


def load_model():
    global num_users, num_movies, embedding_dim

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    num_users = checkpoint["n_users"]
    num_movies = checkpoint["n_items"]
    embedding_dim = checkpoint["embedding_dim"]

    m = MatrixFactorization(num_users, num_movies, embedding_dim).to(device)
    m.load_state_dict(checkpoint["model_state_dict"])
    m.eval()
    return m


@app.on_event("startup")
async def startup_event():
    global model
    model = load_model()
    print("✅ Model loaded on", device)


@app.post("/predict")
async def predict(req: PredictRequest):
    """
    특정 user_idx, movie_idx에 대한 평점 예측
    """
    with torch.no_grad():
        u = torch.LongTensor([req.user_idx]).to(device)
        m_idx = torch.LongTensor([req.movie_idx]).to(device)
        pred = model(u, m_idx).item()

    return {
        "user_idx": req.user_idx,
        "movie_idx": req.movie_idx,
        "prediction": float(pred),
    }

# 69716 (test용)
@app.get("/recommend/{user_idx}")
async def recommend(user_idx: int, top_k: int = 10):


    """
    user_idx에 대해 top_k 개 영화 추천 (movie_idx 리스트 반환)
    """

    print("num_movies:", num_movies)
    print("user embedding size:", model.users_embedding.num_embeddings)
    print("item embedding size:", model.items_embedding.num_embeddings)

    print("requested user:", user_idx)

    if num_movies is None:
        return {"error": "Model not loaded properly"}

    with torch.no_grad():
        movies = torch.arange(num_movies, dtype=torch.long, device=device)
        users = torch.full((num_movies,), user_idx, dtype=torch.long, device=device)
        scores = model(users, movies)

        top_scores, top_indices = torch.topk(scores, top_k)

    return {
        "user_idx": user_idx,
        "top_k": top_k,
        "movie_indices": top_indices.cpu().tolist(),
        "scores": top_scores.cpu().tolist(),
    }