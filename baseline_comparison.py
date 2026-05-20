"""
Baseline Feature Comparison: HuBERT vs Traditional MFCCs
---------------------------------------------------------
Compares your HuBERT+GAP features against traditional MFCC baseline.
This validates your feature extraction approach.
"""

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# CONFIGURATION
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "processed_clips")
CSV_PATH = os.path.join(BASE_DIR, "processed_labels.csv")

# MFCC parameters
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

def extract_mfcc_features(audio_path, max_length=48000):
    """
    Extract traditional MFCC features from audio.
    Returns 40-dimensional MFCC + delta + delta-delta = 120 features
    """
    try:
        # Load audio
        y, sr = sf.read(audio_path)
        
        # Stereo to mono
        if len(y.shape) > 1:
            y = y[:, 0]
        
        # Resample to 16kHz if needed
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Pad or truncate to fixed length (3 seconds)
        if len(y) < max_length:
            y = np.pad(y, (0, max_length - len(y)))
        else:
            y = y[:max_length]
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=N_MFCC, 
            n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        
        # Delta and delta-delta
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Aggregate: mean and std across time
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        delta_mean = np.mean(mfcc_delta, axis=1)
        delta_std = np.std(mfcc_delta, axis=1)
        delta2_mean = np.mean(mfcc_delta2, axis=1)
        delta2_std = np.std(mfcc_delta2, axis=1)
        
        # Concatenate all features
        features = np.concatenate([
            mfcc_mean, mfcc_std,
            delta_mean, delta_std,
            delta2_mean, delta2_std
        ])
        
        return features
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def load_dataset_paths(csv_path):
    """Load file paths and labels from CSV"""
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
    
    return file_paths, labels

def extract_all_mfcc_features():
    """Extract MFCC features for entire dataset"""
    print("="*60)
    print(" EXTRACTING MFCC BASELINE FEATURES")
    print("="*60)
    
    file_paths, labels = load_dataset_paths(CSV_PATH)
    print(f"\nTotal files to process: {len(file_paths)}")
    
    X_mfcc = []
    y_mfcc = []
    
    for path, label in tqdm(zip(file_paths, labels), total=len(file_paths), desc="Extracting MFCCs"):
        features = extract_mfcc_features(path)
        if features is not None:
            X_mfcc.append(features)
            y_mfcc.append(label)
    
    X_mfcc = np.array(X_mfcc, dtype=np.float32)
    y_mfcc = np.array(y_mfcc, dtype=np.int32)
    
    print(f"\nMFCC Features Shape: {X_mfcc.shape}")
    print(f"Labels Shape: {y_mfcc.shape}")
    
    # Save
    np.save("mfcc_features.npy", X_mfcc)
    np.save("mfcc_labels.npy", y_mfcc)
    print("\n[✓] MFCC features saved: mfcc_features.npy")
    
    return X_mfcc, y_mfcc

def compare_features():
    """Compare HuBERT vs MFCC features using XGBoost"""
    print("\n" + "="*60)
    print(" FEATURE COMPARISON: HuBERT vs MFCC")
    print("="*60)
    
    # Load HuBERT features
    if os.path.exists("selected_features.npy"):
        X_hubert = np.load("selected_features.npy")
        y_hubert = np.load("selected_labels.npy")
        print(f"\n[✓] HuBERT features loaded: {X_hubert.shape}")
    else:
        print("\n[ERROR] HuBERT features not found. Run Gap_feature.py first.")
        return
    
    # Load or extract MFCC features
    if os.path.exists("mfcc_features.npy"):
        X_mfcc = np.load("mfcc_features.npy")
        y_mfcc = np.load("mfcc_labels.npy")
        print(f"[✓] MFCC features loaded: {X_mfcc.shape}")
    else:
        print("\n[!] MFCC features not found. Extracting...")
        X_mfcc, y_mfcc = extract_all_mfcc_features()
    
    # Ensure same samples
    min_samples = min(len(X_hubert), len(X_mfcc))
    X_hubert = X_hubert[:min_samples]
    y_hubert = y_hubert[:min_samples]
    X_mfcc = X_mfcc[:min_samples]
    y_mfcc = y_mfcc[:min_samples]
    
    # Split data
    X_h_train, X_h_test, y_h_train, y_h_test = train_test_split(
        X_hubert, y_hubert, test_size=0.2, stratify=y_hubert, random_state=42
    )
    X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(
        X_mfcc, y_mfcc, test_size=0.2, stratify=y_mfcc, random_state=42
    )
    
    # Standardize
    scaler_h = StandardScaler()
    X_h_train = scaler_h.fit_transform(X_h_train)
    X_h_test = scaler_h.transform(X_h_test)
    
    scaler_m = StandardScaler()
    X_m_train = scaler_m.fit_transform(X_m_train)
    X_m_test = scaler_m.transform(X_m_test)
    
    # Train XGBoost on both
    print("\n" + "-"*60)
    print("Training XGBoost on HuBERT features...")
    xgb_hubert = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        random_state=42, n_jobs=-1, eval_metric='logloss'
    )
    xgb_hubert.fit(X_h_train, y_h_train)
    
    y_pred_h = xgb_hubert.predict(X_h_test)
    y_prob_h = xgb_hubert.predict_proba(X_h_test)[:, 1]
    
    acc_h = accuracy_score(y_h_test, y_pred_h)
    f1_h = f1_score(y_h_test, y_pred_h, average='macro')
    auc_h = roc_auc_score(y_h_test, y_prob_h)
    
    print("\n" + "-"*60)
    print("Training XGBoost on MFCC features...")
    xgb_mfcc = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        random_state=42, n_jobs=-1, eval_metric='logloss'
    )
    xgb_mfcc.fit(X_m_train, y_m_train)
    
    y_pred_m = xgb_mfcc.predict(X_m_test)
    y_prob_m = xgb_mfcc.predict_proba(X_m_test)[:, 1]
    
    acc_m = accuracy_score(y_m_test, y_pred_m)
    f1_m = f1_score(y_m_test, y_pred_m, average='macro')
    auc_m = roc_auc_score(y_m_test, y_prob_m)
    
    # Results
    print("\n" + "="*60)
    print(" COMPARISON RESULTS")
    print("="*60)
    print(f"\n{'Metric':<20} {'HuBERT':<15} {'MFCC':<15} {'Improvement':<15}")
    print("-"*60)
    print(f"{'Accuracy':<20} {acc_h:<15.4f} {acc_m:<15.4f} {(acc_h-acc_m)/acc_m*100:>+.2f}%")
    print(f"{'F1-Score (Macro)':<20} {f1_h:<15.4f} {f1_m:<15.4f} {(f1_h-f1_m)/f1_m*100:>+.2f}%")
    print(f"{'ROC-AUC':<20} {auc_h:<15.4f} {auc_m:<15.4f} {(auc_h-auc_m)/auc_m*100:>+.2f}%")
    print("="*60)
    
    # Detailed reports
    print("\n>>> HuBERT Classification Report:")
    print(classification_report(y_h_test, y_pred_h, target_names=['Fluent', 'Stutter'], digits=4))
    
    print("\n>>> MFCC Classification Report:")
    print(classification_report(y_m_test, y_pred_m, target_names=['Fluent', 'Stutter'], digits=4))
    
    # Save results
    results = {
        'hubert': {'accuracy': float(acc_h), 'f1': float(f1_h), 'auc': float(auc_h)},
        'mfcc': {'accuracy': float(acc_m), 'f1': float(f1_m), 'auc': float(auc_m)},
        'improvement': {
            'accuracy': float((acc_h-acc_m)/acc_m*100),
            'f1': float((f1_h-f1_m)/f1_m*100),
            'auc': float((auc_h-auc_m)/auc_m*100)
        }
    }
    
    import json
    with open('feature_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n[✓] Results saved: feature_comparison_results.json")

if __name__ == "__main__":
    compare_features()
