"""
HuBERT GAP Embedding → SelectKBest Feature Selection
----------------------------------------------------

Bu kod:
- embedding_features.npy yükler
- embedding_labels.npy yükler
- En iyi K feature'ı seçer
- Yeni feature matrisini kaydeder

ÇIKTI:
selected_features.npy
selected_labels.npy
"""

import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# =====================================================
# AYARLAR
# =====================================================

K_FEATURES = 256

# =====================================================
# EMBEDDINGLERİ YÜKLE
# =====================================================

print("Embedding dosyaları yükleniyor...")

X = np.load("embedding_features.npy")
y = np.load("embedding_labels.npy")

print(f"\nOrijinal Shape:")
print(f"X: {X.shape}")
print(f"y: {y.shape}")

# =====================================================
# SELECT KBEST
# =====================================================

print(f"\nEn iyi {K_FEATURES} feature seçiliyor...")

selector = SelectKBest(
    score_func=mutual_info_classif,
    k=K_FEATURES
)

X_selected = selector.fit_transform(X, y)

# =====================================================
# SONUÇLAR
# =====================================================

print("\nFeature Selection Tamamlandı")
print(f"Yeni Shape: {X_selected.shape}")

# =====================================================
# FEATURE SCORE RAPORU
# =====================================================

scores = selector.scores_

top_indices = np.argsort(scores)[::-1][:20]

print("\nEn güçlü ilk 20 feature:\n")

for rank, idx in enumerate(top_indices, start=1):
    print(
        f"{rank:2d}. Feature {idx:3d}  |  Score: {scores[idx]:.6f}"
    )

# =====================================================
# KAYDET
# =====================================================

np.save("selected_features.npy", X_selected)
np.save("selected_labels.npy", y)

print("\nKaydedildi:")
print("selected_features.npy")
print("selected_labels.npy")