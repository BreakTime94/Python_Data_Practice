import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import joblib
import json  # ✅ Epoch 정보를 저장하기 위한 JSON
import warnings
warnings.filterwarnings("ignore")
from bertClassification import BERTClassifier, WebtoonDataset

# ✅ 3. 추가 학습이 가능하도록 구현
def continue_training(df, path, epochs_to_train=10):

    if os.path.exists(path):
        df_new = pd.read_excel(path, engine="openpyxl")
        df = pd.concat([df, df_new], ignore_index=True)  # ✅ 기존 데이터와 합치기
        print("✅ 새로운 데이터 추가 완료!")

    # ✅ 저장 경로 설정
    local_model_dir = os.path.join(os.path.dirname(__file__), "classifier")
    os.makedirs(local_model_dir, exist_ok=True)

    label_encoder_path = os.path.join(local_model_dir, "label_encoder.pkl")
    model_path = os.path.join(local_model_dir, "bert_webtoon_classifier.pth")
    epoch_info_path = os.path.join(local_model_dir, "epoch_info.json")  # ✅ Epoch 정보 저장 파일

    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    # ✅ 기존 라벨 인코더 로드 또는 새로 생성
    if os.path.exists(label_encoder_path):
        label_encoder = joblib.load(label_encoder_path)
        print(f"✅ 기존 라벨 인코더 로드 완료: {label_encoder_path}")
    else:
        label_encoder = LabelEncoder()
        print("🚀 새로운 라벨 인코더 생성")

    df["genre"] = df["genre"].apply(lambda x: eval(x)[0])
    df["labels"] = label_encoder.fit_transform(df["genre"])

    # ✅ 라벨 인코더 저장
    joblib.dump(label_encoder, label_encoder_path)
    print(f"✅ 라벨 인코더 저장 완료: {label_encoder_path}")

    # ✅ 데이터 분할
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["synopsis"].tolist(), df["labels"].tolist(), test_size=0.2, random_state=42
    )

    train_dataset = WebtoonDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    test_dataset = WebtoonDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)  # ✅ 테스트용 데이터 로더 추가

    # ✅ Cuda, MPS, Cpu 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "mps"
    if torch.backends.mps.is_available() else "cpu")
    print(f"✅ 실행 장치: {device}")

    num_classes = len(label_encoder.classes_)

    # ✅ 모델 로드
    model = BERTClassifier(num_classes=num_classes).to(device)
    start_epoch = 0  # ✅ 학습 시작 epoch 기본값

    # ✅ 기존 모델이 있다면 불러오기
    if os.path.exists(model_path):
        print(f"✅ 기존 모델 로드 중... ({model_path})")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ 기존 모델 로드 완료!")

        # ✅ 기존 Epoch 정보 불러오기
        if os.path.exists(epoch_info_path):
            with open(epoch_info_path, "r") as f:
                epoch_info = json.load(f)
                start_epoch = epoch_info.get("last_epoch", 0)  # ✅ 마지막 학습된 epoch 불러오기
                print(f"🔄 기존 학습 Epoch: {start_epoch}")
        else:
            print("⚠️ Epoch 정보 없음. 처음부터 학습을 진행합니다.")
    else:
        print("🚀 새 모델 학습 시작!")

    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # ✅ 추가 학습 루프
    for epoch in tqdm(range(start_epoch, start_epoch + epochs_to_train)):
        model.train()
        total_loss = 0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1}/{start_epoch + epochs_to_train}], Learning Rate: {current_lr:.6f}")

        for input_ids, attention_mask, labels in train_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{start_epoch + epochs_to_train}], Loss: {total_loss / len(train_loader):.4f}")

        # ✅ 모델 저장
        torch.save(model.state_dict(), model_path)

        # ✅ 학습된 Epoch 정보 저장
        epoch_info = {"last_epoch": epoch + 1}
        with open(epoch_info_path, "w") as f:
            json.dump(epoch_info, f)

        print(f"✅ 모델 저장 완료 (Epoch: {epoch + 1})")

    print("🎉 추가 학습 완료!")

    # ✅ 학습 후 테스트 정확도 출력
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"✅ 테스트 정확도: {accuracy:.2f}%")  # ✅ 정확도 출력

# ✅ 실행
if __name__ == "__main__":
    # ✅ 기존 데이터 로드
    df = pd.read_excel("../data/NAVER-Webtoon_OSMU.xlsx", engine="openpyxl")

    # ✅ 새로운 데이터 추가
    new_data_path = "../data/new_data.xlsx"
    continue_training(df, new_data_path, epochs_to_train=100)  # 원하는 추가 학습 횟수를 변경 가능