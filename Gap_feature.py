"""
HuBERT GAP Embedding Extraction
--------------------------------
Bu kod:
- Fine-tuned HuBERT modelini yükler
- Her ses dosyasından embedding çıkarır
- Global Average Pooling (GAP) uygular
- 768 boyutlu feature üretir
- NPY olarak kaydeder

ÇIKTI:
embedding_features.npy
embedding_labels.npy

Sonrasında:
XGBoost / SVM / LightGBM / CatBoost için kullanabilirsin.
"""

import os
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel
)

# =====================================================
# AYARLAR
# =====================================================

BASE_DIR = r"C:\Users\BIL MUH\Desktop\Damla\stutter\archive"

AUDIO_DIR = os.path.join(BASE_DIR, "processed_clips")

CSV_PATH = os.path.join(BASE_DIR, "train_split.csv")

# Fine-tuned model klasörün
MODEL_PATH = os.path.join(BASE_DIR, "best_hubert_stuttering_model")

MAX_LENGTH = 16000 * 3   # 3 saniye

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# FEATURE EXTRACTOR + MODEL
# =====================================================

print("HuBERT modeli yükleniyor...")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)

model = HubertModel.from_pretrained(MODEL_PATH)

model.to(DEVICE)
model.eval()

print(f"Kullanılan cihaz: {DEVICE}")

# =====================================================
# DATASET YÜKLEME
# =====================================================

def load_dataset(csv_path):

    df = pd.read_csv(csv_path)

    file_paths = []
    labels = []

    for _, row in df.iterrows():

        show = str(row["Show"]).strip()
        ep_id = str(row["EpId"]).strip()
        clip_id = str(row["ClipId"]).strip()

        if "fluencybank" in show.lower():
            ep_id = ep_id.zfill(3)

        filename = f"{show}_{ep_id}_{clip_id}.wav"

        full_path = os.path.join(AUDIO_DIR, filename)

        if os.path.exists(full_path):

            file_paths.append(full_path)
            labels.append(int(row["is_stutter"]))

    print(f"Toplam geçerli dosya: {len(file_paths)}")

    return file_paths, labels

# =====================================================
# EMBEDDING ÇIKARMA (GAP)
# =====================================================

def extract_gap_embedding(audio_path):

    try:
        speech, sr = sf.read(audio_path)

        # stereo → mono
        if len(speech.shape) > 1:
            speech = speech[:, 0]

        # çok kısa / bozuk kontrolü
        if len(speech) < 400:
            return None

    except:
        return None

    # Feature extraction
    inputs = feature_extractor(
        speech,
        sampling_rate=16000,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_attention_mask=True
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():

        outputs = model(
            input_values=inputs["input_values"],
            attention_mask=inputs["attention_mask"]
        )

        # last hidden state
        hidden_states = outputs.last_hidden_state
        # shape:
        # (batch, time, hidden)

        # =================================================
        # GLOBAL AVERAGE POOLING (GAP)
        # =================================================

        embedding = torch.mean(hidden_states, dim=1)

        embedding = embedding.squeeze().cpu().numpy()

    return embedding

# =====================================================
# TÜM DATASET İÇİN EMBEDDING ÜRET
# =====================================================

print("\nEmbedding çıkarılıyor...\n")

file_paths, labels = load_dataset(CSV_PATH)

X = []
y = []

for path, label in tqdm(zip(file_paths, labels), total=len(file_paths)):

    emb = extract_gap_embedding(path)

    if emb is None:
        continue

    X.append(emb)
    y.append(label)

# =====================================================
# NUMPY ARRAY
# =====================================================

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print("\nEmbedding Shape:", X.shape)
print("Label Shape:", y.shape)

# =====================================================
# KAYDET
# =====================================================

np.save("embedding_features.npy", X)
np.save("embedding_labels.npy", y)

print("\nKaydedildi:")
print("embedding_features.npy")
print("embedding_labels.npy")