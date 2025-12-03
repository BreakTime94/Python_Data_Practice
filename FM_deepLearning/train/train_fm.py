import pandas as pd
import torch, os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.FM_recommend import MatrixFactorization, device, evaluate, extract_ratings, RatingsDataset

MODEL_PATH = "../model"

def learning_train(user_ids, item_ids, train_loader, test_data, learning_rate, vector_dim, epochs):
  n_users = len(user_ids)
  n_items = len(item_ids)

  model = MatrixFactorization(n_users, n_items, vector_dim).to(device)
  loss_fn = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # backword (역전파: 미분)


  for epochs in range(epochs):
    model.train()
    total_loss = 0.0
    batch_tensor = tqdm(train_loader, desc=f"Epoch {epochs}/{epochs}", leave=False)

    for user , item , rating in batch_tensor:
      user = user.to(device)
      item = item.to(device)
      rating = rating.float().to(device)

      optimizer.zero_grad()
      pred = model(user, item)
      loss = loss_fn(pred, rating)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()

      batch_tensor.set_postfix(loss=loss.item())

    print(f"Epoch[{epochs}/{epochs}], Loss: {total_loss: .4f}]")

  evaluate(model, test_data, device=device)
  # 모델 저장
  os.makedirs(f"{MODEL_PATH}", exist_ok=True)
  save_path = os.path.join(MODEL_PATH, "fm_model.pt")
  torch.save({
    "model_state_dict": model.state_dict(),
    "n_users": n_users,
    "n_items": n_items,
    "embedding_dim": vector_dim,
    "user_idx":user_idx,
    "item_idx":item_idx
  },
  f"{save_path}"
  )
  print(f"Model 저장 완료 -> {MODEL_PATH}")

if __name__ == "__main__":
  vector_dim = 64 # 2의 배수로 가면 됨
  epochs = 10
  learning_rate = 0.001
  batch_size = 64
  df = pd.read_pickle("../data/ratings.pkl")
  df = extract_ratings(df)
  user_ids = df["user_id"].unique()
  item_ids = df["movie_id"].unique()

  user_idx = {u: i for i , u in enumerate(user_ids)}
  item_idx = {item: i for i , item in enumerate(item_ids)}
  df["user_idx"] = df["user_id"].map(user_idx)
  df["item_idx"] = df["movie_id"].map(item_idx)

  train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

  # print(test_data)
  # exit()

  train_loader = RatingsDataset(
    torch.LongTensor(train_data["user_idx"].values),
    torch.LongTensor(train_data["item_idx"].values),
    torch.LongTensor(train_data["rating"].values),
  )
  train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)

  learning_train(
    user_ids,
    item_ids,
    train_loader,
    test_data,
    learning_rate,
    vector_dim,
    epochs,
  )
