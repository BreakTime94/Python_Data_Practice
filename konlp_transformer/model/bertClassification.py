import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import joblib
import json
 # Epoch 정보를 저장하기 위한 JSON


 # ✅ 1. BERT 분류 모델 정의
class BERTClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.fc(self.dropout(pooled_output))


# ✅ 2. PyTorch Dataset 정의
class WebtoonDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        tokens = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_len,
                                return_tensors="pt")

        input_ids = tokens["input_ids"].squeeze(0)  # (1, max_len) → (max_len)
        attention_mask = tokens["attention_mask"].squeeze(0)

        return input_ids, attention_mask, label

