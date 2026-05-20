"""
===============================================================================
STUTTERING DETECTION — 5 MODEL OOF ENSEMBLE (PRODUCTION-READY v3)
===============================================================================

METODOLOJİ:
  • Veri bölme    : %80 train / %20 test (sabit, stratifiye)
  • OOF yapısı    : %80'lik train üzerinde Stratified K-Fold (k=5)
                    → Her fold kendi val parçasında izole early stopping
                    → Her fold'un best_iteration / best_epoch'u kaydedilir
                    → OOF olasılıkları sızıntısız biriktirilir
  • Ağırlık araması : OOF olasılıkları üzerinde ROC-AUC maksimizasyonu
                      (eşikten bağımsız metrik → threshold tutarsızlığı yok)
  • Optimal eşik  : Youden Index ile OOF blended olasılıkları üzerinde
  • Final eğitim  : Her model, OOF fold'larından hesaplanan ORTALAMA
                    best_iteration / best_epoch ile SABIT adımda eğitilir.
                    → OOF'taki ve final'daki model karmaşıklığı eşleşir.
  • Final rapor   : Test seti sadece burada, tek kez kullanılır

DÜZELTMELER:
  1. Validasyon sızıntısı    : OOF kurgusuyla giderildi
  2. Eşik tutarsızlığı       : AUC opt. + Youden Index ile giderildi
  3. Val_size hatası          : K-Fold yapısıyla geçersiz kılındı
  4. XGBoost early_stopping   : constructor'a taşındı
  5. Model karmaşıklığı uyumsuzluğu : OOF ortalama iter/epoch ile giderildi ← YENİ
===============================================================================
"""

import os
import warnings
import json

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import joblib
from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve,
    precision_recall_curve,
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    Dense, Dropout, BatchNormalization,
    Bidirectional, LSTM, Input,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt

# ==============================================================================
# 1. KONFİGÜRASYON
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CFG = {
    "features_path": os.path.join(SCRIPT_DIR, "selected_features.npy"),
    "labels_path":   os.path.join(SCRIPT_DIR, "selected_labels.npy"),
    "output_dir":    os.path.join(SCRIPT_DIR, "ensemble_outputs"),

    # Veri bölme
    "test_size":   0.20,
    "random_seed": 42,
    "n_folds":     5,       # OOF için K-Fold sayısı

    # XGBoost — early_stopping_rounds artık constructor'da
    "xgb_params": dict(
        n_estimators=1000,          # early stopping ile dinamik kesilir
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=30,  # ← constructor'da (eleştiri #4)
        random_state=42,
        n_jobs=-1,
    ),
    "svm_params": dict(
        kernel="rbf", C=1.0, probability=True,
        class_weight="balanced", random_state=42,
    ),
    "rf_params": dict(
        n_estimators=200, max_depth=None,
        class_weight="balanced", n_jobs=-1, random_state=42,
    ),

    # Derin öğrenme
    "epochs":     60,
    "batch_size": 32,
    "patience":   10,

    # Ağırlık araması
    "n_iterations": 5000,
    # OOF üzerinde ROC-AUC maksimize edilir (eşiksiz → tutarsızlık yok)
    "search_metric": "auc",
}

MODEL_NAMES = ["XGBoost", "SVM", "RandomForest", "CNN", "BiLSTM"]
CLASS_NAMES  = ["Akici(0)", "Kekeme(1)"]
os.makedirs(CFG["output_dir"], exist_ok=True)


# ==============================================================================
# 2. VERİ YÜKLEME & ANA BÖLME
# ==============================================================================
def load_data():
    X = np.load(CFG["features_path"]).astype(np.float32)
    y = np.load(CFG["labels_path"]).astype(np.int32)
    return X, y


def split_train_test(X, y):
    """
    Tek seferlik %80/%20 bölme.
    Test seti buradan sonra hiçbir yerde kullanılmaz — sadece final raporda.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CFG["test_size"],
        stratify=y,
        random_state=CFG["random_seed"],
    )
    print(f"\n  Veri Dağılımı:")
    print(f"    Train : {len(y_train)}  "
          f"(neg={( y_train==0).sum()}, pos={(y_train==1).sum()})")
    print(f"    Test  : {len(y_test)}")
    return X_train, X_test, y_train, y_test


# ==============================================================================
# 3. MODEL MİMARİLERİ
# ==============================================================================
def build_cnn(n_features):
    model = Sequential([
        Input(shape=(n_features, 1)),
        Conv1D(64,  kernel_size=5, activation="relu", padding="same"),
        BatchNormalization(), MaxPooling1D(2), Dropout(0.25),
        Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(), MaxPooling1D(2), Dropout(0.25),
        Conv1D(256, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(), GlobalAveragePooling1D(), Dropout(0.40),
        Dense(128, activation="relu"), Dropout(0.30),
        Dense(1,   activation="sigmoid"),
    ], name="CNN_1D")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy", metrics=["accuracy"],
    )
    return model


def build_bilstm(n_features):
    model = Sequential([
        Input(shape=(n_features, 1)),
        Bidirectional(LSTM(128, return_sequences=True)), Dropout(0.30),
        Bidirectional(LSTM(64,  return_sequences=False)), Dropout(0.30),
        Dense(64, activation="relu"), Dropout(0.25),
        Dense(1,  activation="sigmoid"),
    ], name="BiLSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss="binary_crossentropy", metrics=["accuracy"],
    )
    return model


def _fresh_callbacks():
    """Her fold için tamamen yeni, izole callback nesneleri."""
    return [
        EarlyStopping(
            monitor="val_loss", patience=CFG["patience"],
            restore_best_weights=True, verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5,
            min_lr=1e-6, verbose=0,
        ),
    ]


# ==============================================================================
# 4. OOF OLASİLIK ÜRETME
# ==============================================================================
def compute_oof_probabilities(X_train, y_train):
    """
    Stratified K-Fold ile her model için Out-of-Fold olasılıkları üretir.

    Her fold'da:
      - Scaler YALNIZCA fold'un train parçasına fit edilir.
      - Val parçası sadece transform edilir → sızıntı yok.
      - XGBoost ve DL modellerinin early stopping'i fold'un val'ına göre yapılır.
      - Her fold'un best_iteration / best_epoch'u kaydedilir.

    Dönüş:
        oof_probas  : liste[ndarray(n_train,)]  — her modelin P(class=1) OOF tahmini
        best_iters  : dict — her model için fold best iter/epoch listesi
                      {"XGBoost": [62, 71, 58, ...], "CNN": [18, 22, 15, ...], ...}
    """
    n = len(y_train)
    n_feat   = X_train.shape[1]

    oof_probas = [np.zeros(n, dtype=np.float32) for _ in range(len(MODEL_NAMES))]

    # Her model için fold bazlı best iter/epoch izleme
    best_iters: dict[str, list[int]] = {name: [] for name in MODEL_NAMES}

    skf = StratifiedKFold(
        n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["random_seed"]
    )

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"\n{'='*55}")
        print(f"  FOLD {fold_idx}/{CFG['n_folds']}")
        print(f"{'='*55}")

        Xf_tr, Xf_val = X_train[tr_idx], X_train[val_idx]
        yf_tr, yf_val = y_train[tr_idx], y_train[val_idx]

        # --- Scaler: sadece fold train'e fit ---
        scaler = StandardScaler()
        Xf_tr_sc  = scaler.fit_transform(Xf_tr).astype(np.float32)
        Xf_val_sc = scaler.transform(Xf_val).astype(np.float32)

        neg = (yf_tr == 0).sum()
        pos = (yf_tr == 1).sum()
        scale_pos_weight = float(neg / pos)

        # ── XGBoost ──────────────────────────────────────────────
        xgb_params = dict(CFG["xgb_params"])
        xgb_params["scale_pos_weight"] = scale_pos_weight
        xgb = XGBClassifier(**xgb_params)
        xgb.fit(
            Xf_tr_sc, yf_tr,
            eval_set=[(Xf_val_sc, yf_val)],
            verbose=False,
        )
        oof_probas[0][val_idx] = xgb.predict_proba(Xf_val_sc)[:, 1]
        best_iters["XGBoost"].append(xgb.best_iteration)
        print(f"  [✓] XGBoost  best_iter={xgb.best_iteration}")
        del xgb

        # ── SVM ──────────────────────────────────────────────────
        # SVM'de iterasyon kavramı yok; sabit hiperparametre → kayıt gereksiz
        svm = SVC(**CFG["svm_params"])
        svm.fit(Xf_tr_sc, yf_tr)
        oof_probas[1][val_idx] = svm.predict_proba(Xf_val_sc)[:, 1]
        print(f"  [✓] SVM")
        del svm

        # ── RandomForest ─────────────────────────────────────────
        # RF'de early stopping yok; n_estimators sabit → kayıt gereksiz
        rf = RandomForestClassifier(**CFG["rf_params"])
        rf.fit(Xf_tr_sc, yf_tr)
        oof_probas[2][val_idx] = rf.predict_proba(Xf_val_sc)[:, 1]
        print(f"  [✓] RandomForest")
        del rf

        # ── CNN & BiLSTM: kanal boyutu ───────────────────────────
        Xf_tr_dl  = Xf_tr_sc[:, :, np.newaxis]
        Xf_val_dl = Xf_val_sc[:, :, np.newaxis]
        cw = {0: 1.0, 1: scale_pos_weight}

        for m_idx, (model_name, build_fn) in enumerate(
            [("CNN", build_cnn), ("BiLSTM", build_bilstm)], start=3
        ):
            tf.keras.backend.clear_session()
            model = build_fn(n_feat)
            history = model.fit(
                Xf_tr_dl, yf_tr,
                validation_data=(Xf_val_dl, yf_val),
                epochs=CFG["epochs"],
                batch_size=CFG["batch_size"],
                callbacks=_fresh_callbacks(),
                class_weight=cw,
                verbose=0,
            )
            # restore_best_weights=True olduğundan tahmin en iyi ağırlıkla yapılır
            best_ep = int(np.argmin(history.history["val_loss"])) + 1
            best_iters[model_name].append(best_ep)
            oof_probas[m_idx][val_idx] = model.predict(
                Xf_val_dl, verbose=0
            ).ravel()
            print(f"  [✓] {model_name:<10} best_epoch={best_ep}")
            del model

    # OOF AUC raporu (sanity check)
    print("\n  OOF ROC-AUC (sızıntısız):")
    for name, proba in zip(MODEL_NAMES, oof_probas):
        auc = roc_auc_score(y_train, proba)
        print(f"    {name:<15} AUC={auc:.4f}")

    # OOF best iter/epoch özeti → final eğitimde kullanılacak
    print("\n  OOF Best Iter / Epoch Özeti:")
    for name, iters in best_iters.items():
        if iters:
            mean_iter = int(round(np.mean(iters)))
            print(f"    {name:<15} fold iter/epoch={iters}  → ortalama={mean_iter}")

    return oof_probas, best_iters


# ==============================================================================
# 5. TEST SETİ İÇİN TAM MODEL EĞİTİMİ
# ==============================================================================
def train_final_models(X_train, y_train, best_iters: dict):
    """
    Tüm train verisiyle modelleri yeniden eğitir.

    Kritik ilke — Model Karmaşıklığı Eşleşmesi:
      OOF ağırlık araması, fold'lardaki modellerin karmaşıklığına göre yapıldı.
      Final modeller de AYNI karmaşıklıkta olmalı; aksi hâlde ensemble ağırlıkları
      artık bu modellere uymaz.

      → XGBoost : n_estimators = round(mean(best_iteration per fold))
                  early_stopping kapalı (eval_set yok), sabit adımda durur.
      → CNN/BiLSTM: epochs = round(mean(best_epoch per fold))
                    early_stopping yok, sabit epoch'ta durur.
      → SVM / RF  : Değişmez; hiperparametreleri zaten sabit.
    """
    print("\n>>> Final Modeller Eğitiliyor (OOF ortalama iter/epoch ile)...")
    n_feat = X_train.shape[1]

    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(X_train).astype(np.float32)
    joblib.dump(scaler, os.path.join(CFG["output_dir"], "scaler_final.pkl"))

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = float(neg / pos)

    models = {}

    # ── XGBoost ─────────────────────────────────────────────────────────────
    mean_xgb_iter = int(round(np.mean(best_iters["XGBoost"])))
    print(f"  XGBoost  → n_estimators={mean_xgb_iter} (OOF ortalaması)")
    # early_stopping_rounds'u silerek sabit adımda eğitiyoruz
    xgb_params_final = {k: v for k, v in CFG["xgb_params"].items()
                        if k != "early_stopping_rounds"}
    xgb_params_final["n_estimators"]     = mean_xgb_iter
    xgb_params_final["scale_pos_weight"] = scale_pos_weight
    xgb = XGBClassifier(**xgb_params_final)
    xgb.fit(Xtr_sc, y_train, verbose=False)
    models["XGBoost"] = xgb
    print("  [✓] XGBoost")

    # ── SVM ─────────────────────────────────────────────────────────────────
    svm = SVC(**CFG["svm_params"])
    svm.fit(Xtr_sc, y_train)
    models["SVM"] = svm
    print("  [✓] SVM")

    # ── RandomForest ────────────────────────────────────────────────────────
    rf = RandomForestClassifier(**CFG["rf_params"])
    rf.fit(Xtr_sc, y_train)
    models["RandomForest"] = rf
    print("  [✓] RandomForest")

    # ── CNN & BiLSTM ────────────────────────────────────────────────────────
    Xtr_dl = Xtr_sc[:, :, np.newaxis]
    cw     = {0: 1.0, 1: scale_pos_weight}

    for model_name, build_fn in [("CNN", build_cnn), ("BiLSTM", build_bilstm)]:
        mean_ep = int(round(np.mean(best_iters[model_name])))
        print(f"  {model_name:<10} → epochs={mean_ep} (OOF ortalaması)")
        tf.keras.backend.clear_session()
        model = build_fn(n_feat)
        # Sabit epoch — early stopping YOK, callbacks YOK
        model.fit(
            Xtr_dl, y_train,
            epochs=mean_ep,
            batch_size=CFG["batch_size"],
            class_weight=cw,
            verbose=0,
        )
        models[model_name] = model
        print(f"  [✓] {model_name}")

    return models, scaler


def predict_test(models, scaler, X_test):
    """Eğitilmiş final modellerden test seti olasılıklarını toplar."""
    Xte_sc = scaler.transform(X_test).astype(np.float32)
    Xte_dl = Xte_sc[:, :, np.newaxis]

    test_probas = []
    for name in ["XGBoost", "SVM", "RandomForest"]:
        test_probas.append(models[name].predict_proba(Xte_sc)[:, 1])
    for name in ["CNN", "BiLSTM"]:
        test_probas.append(models[name].predict(Xte_dl, verbose=0).ravel())

    return test_probas   # liste[ndarray(n_test,)]


# ==============================================================================
# 6. AĞIRLIK ARAMAĞI (OOF, AUC MAKSİMİZASYONU)
# ==============================================================================
def weighted_blend(probas, weights):
    """P(class=1) ağırlıklı karışımı. probas: liste[ndarray(n,)]"""
    return sum(w * p for w, p in zip(weights, probas))


def search_best_weights(oof_probas, y_train):
    """
    OOF olasılıkları üzerinde ROC-AUC maksimize edilir.
    Eşik kullanılmaz → threshold tutarsızlığı sıfır.
    """
    print(f"\n>>> Ağırlık Araması (OOF, AUC, {CFG['n_iterations']} iter)...")
    n_models = len(oof_probas)
    rng      = np.random.default_rng(CFG["random_seed"])

    best = {"weights": None, "auc": -np.inf, "iteration": -1}

    with tqdm(total=CFG["n_iterations"], desc="Ağırlık Arama",
              unit="iter", colour="cyan") as pbar:
        for i in range(1, CFG["n_iterations"] + 1):
            weights = rng.dirichlet(np.ones(n_models, dtype=np.float32))
            blended = weighted_blend(oof_probas, weights)
            auc     = roc_auc_score(y_train, blended)

            if auc > best["auc"]:
                best.update({
                    "weights":   weights.astype(np.float32).copy(),
                    "auc":       auc,
                    "iteration": i,
                })
            pbar.update(1)

    print(f"\n  En iyi OOF AUC : {best['auc']:.4f}  "
          f"(iter {best['iteration']})")
    return best


# ==============================================================================
# 7. YOUDEN INDEX İLE OPTİMAL EŞİK
# ==============================================================================
def find_optimal_threshold(oof_probas, y_train, weights):
    """
    Youden Index = Sensitivity + Specificity - 1
    ROC eğrisinin tüm eşik noktaları üzerinde maksimize edilir.
    OOF olasılıkları üzerinde hesaplanır → test'e sızmaz.
    """
    blended = weighted_blend(oof_probas, weights)
    fpr, tpr, thresholds = roc_curve(y_train, blended)
    youden = tpr - fpr                         # = Sensitivity + Specificity - 1
    best_idx   = int(np.argmax(youden))
    opt_thresh = float(thresholds[best_idx])

    print(f"\n  Optimal Eşik (Youden Index): {opt_thresh:.4f}")
    print(f"    Sensitivity (TPR) : {tpr[best_idx]:.4f}")
    print(f"    Specificity (1-FPR): {1 - fpr[best_idx]:.4f}")
    return opt_thresh


# ==============================================================================
# 8. RAPORLAMA & GÖRSELLEŞTİRME
# ==============================================================================
def print_final_report(best, optimal_threshold, test_probas, y_test):
    blended = weighted_blend(test_probas, best["weights"])
    y_pred  = (blended >= optimal_threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    auc = roc_auc_score(y_test, blended)

    print("\n" + "=" * 60)
    print("  HİBRİT ENSEMBLE — NİHAİ TEST RAPORU")
    print("=" * 60)
    print(f"  Optimal Eşik (Youden)  : {optimal_threshold:.4f}")
    print(f"  Test Accuracy          : {acc:.4f}")
    print(f"  Test F1 (Macro)        : {f1:.4f}")
    print(f"  Test ROC-AUC           : {auc:.4f}\n")

    print(f"  {'Model':<18} {'OOF Ağırlık':>12}")
    print(f"  {'-' * 32}")
    for name, w in zip(MODEL_NAMES, best["weights"]):
        print(f"  {name:<18} {w:>12.4f}")

    print("\n>>> Detaylı Sınıflandırma Raporu:")
    print(classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0
    ))

    # Kaydet
    results = {
        "weights":           {n: float(w) for n, w in zip(MODEL_NAMES, best["weights"])},
        "optimal_threshold": optimal_threshold,
        "oof_auc":           float(best["auc"]),
        "test_accuracy":     float(acc),
        "test_f1_macro":     float(f1),
        "test_roc_auc":      float(auc),
    }
    out = os.path.join(CFG["output_dir"], "best_weights.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Sonuçlar kaydedildi: {out}")
    return y_pred, blended


def plot_all(y_test, y_pred, y_prob, optimal_threshold):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Confusion Matrix (ham)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=axes[0], colorbar=False, cmap="Blues"
    )
    axes[0].set_title(f"Confusion Matrix (eşik={optimal_threshold:.3f})")

    # Confusion Matrix (normalize)
    cm_n = confusion_matrix(y_test, y_pred, normalize="true")
    ConfusionMatrixDisplay(cm_n, display_labels=CLASS_NAMES).plot(
        ax=axes[1], colorbar=False, cmap="Greens"
    )
    axes[1].set_title("Normalize Confusion Matrix")

    # ROC Eğrisi
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    axes[2].plot(fpr, tpr, lw=2, label=f"ROC (AUC={auc:.4f})")
    axes[2].plot([0, 1], [0, 1], "k--", lw=1)
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].set_title("ROC Eğrisi (Test Seti)")
    axes[2].legend()

    plt.tight_layout()
    out = os.path.join(CFG["output_dir"], "ensemble_results.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"  Grafik kaydedildi: {out}")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 60)
    print("  KEKEMELIK TESPİT — OOF HİBRİT ENSEMBLE v3")
    print("=" * 60)

    # 1. Veri yükle & böl
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    del X

    # 2. OOF olasılıkları + fold bazlı best iter/epoch — test seti kullanılmaz
    oof_probas, best_iters = compute_oof_probabilities(X_train, y_train)

    # 3. OOF üzerinde ağırlık araması (AUC, eşiksiz)
    best = search_best_weights(oof_probas, y_train)

    # 4. OOF üzerinde optimal eşik (Youden Index)
    optimal_threshold = find_optimal_threshold(oof_probas, y_train, best["weights"])

    # 5. Final modelleri OOF ortalama iter/epoch ile eğit
    models, scaler = train_final_models(X_train, y_train, best_iters)

    # 6. Test seti tahmini — sadece burada, tek kez
    test_probas = predict_test(models, scaler, X_test)

    # 7. Rapor & görselleştirme
    y_pred, y_prob = print_final_report(best, optimal_threshold, test_probas, y_test)
    plot_all(y_test, y_pred, y_prob, optimal_threshold)


if __name__ == "__main__":
    main()
