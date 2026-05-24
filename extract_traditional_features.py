"""
Traditional Audio Feature Extraction for Stuttering Detection
Uses MFCC, spectral features, and prosodic features
No deep learning model required
"""

import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "processed_clips")
TRAIN_CSV = os.path.join(BASE_DIR, "train_split.csv")
VAL_CSV = os.path.join(BASE_DIR, "val_split.csv")
TEST_CSV = os.path.join(BASE_DIR, "test_split.csv")

def extract_features(audio_path, sr=16000):
    """
    Extract traditional audio features from a single audio file.
    Returns a feature vector of ~40 dimensions.
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=sr, duration=3.0)
        
        # Handle stereo
        if len(y.shape) > 1:
            y = y[:, 0]
        
        # Check if audio is too short or silent
        if len(y) < 400 or np.max(np.abs(y)) < 0.001:
            return None
        
        features = []
        
        # 1. MFCCs (13 coefficients) - most important for speech
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        features.extend(mfccs_mean)  # 13 features
        features.extend(mfccs_std)   # 13 features
        
        # 2. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.append(np.mean(spectral_centroids))
        features.append(np.std(spectral_centroids))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features.append(np.mean(spectral_rolloff))
        features.append(np.std(spectral_rolloff))
        
        # 3. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        
        # 4. RMS Energy
        rms = librosa.feature.rms(y=y)[0]
        features.append(np.mean(rms))
        features.append(np.std(rms))
        
        # 5. Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.append(np.mean(chroma))
        features.append(np.std(chroma))
        
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def load_and_extract(csv_path, split_name):
    """Load CSV and extract features for all audio files."""
    print(f"\nProcessing {split_name} split...")
    
    df = pd.read_csv(csv_path)
    
    features_list = []
    labels_list = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {split_name}"):
        show = str(row['Show']).strip()
        ep_id = str(row['EpId']).strip()
        clip_id = str(row['ClipId']).strip()
        
        # Handle FluencyBank naming
        if "fluencybank" in show.lower():
            ep_id = ep_id.zfill(3)
        
        filename = f"{show}_{ep_id}_{clip_id}.wav"
        filepath = os.path.join(AUDIO_DIR, filename)
        
        if not os.path.exists(filepath):
            continue
        
        features = extract_features(filepath)
        
        if features is not None:
            features_list.append(features)
            labels_list.append(int(row['is_stutter']))
    
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)
    
    print(f"{split_name} - Features shape: {X.shape}, Labels shape: {y.shape}")
    
    return X, y

def main():
    print("="*60)
    print(" TRADITIONAL FEATURE EXTRACTION ")
    print("="*60)
    print("\nExtracting MFCC and spectral features...")
    print("This will take 30-60 minutes for all splits.\n")
    
    # Extract features for all splits
    X_train, y_train = load_and_extract(TRAIN_CSV, "Train")
    X_val, y_val = load_and_extract(VAL_CSV, "Validation")
    X_test, y_test = load_and_extract(TEST_CSV, "Test")
    
    # Combine train and val for full training set
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    
    print("\n" + "="*60)
    print(" SAVING FEATURES ")
    print("="*60)
    
    # Save for hybrid.py (expects selected_features.npy and selected_labels.npy)
    np.save("selected_features.npy", X_full)
    np.save("selected_labels.npy", y_full)
    print(f"✓ Saved: selected_features.npy {X_full.shape}")
    print(f"✓ Saved: selected_labels.npy {y_full.shape}")
    
    # Also save test set separately
    np.save("test_features.npy", X_test)
    np.save("test_labels.npy", y_test)
    print(f"✓ Saved: test_features.npy {X_test.shape}")
    print(f"✓ Saved: test_labels.npy {y_test.shape}")
    
    print("\n" + "="*60)
    print(" FEATURE EXTRACTION COMPLETE ")
    print("="*60)
    print(f"\nTotal training samples: {len(X_full)}")
    print(f"Total test samples: {len(X_test)}")
    print(f"Feature dimensions: {X_full.shape[1]}")
    print(f"\nClass distribution (train):")
    print(f"  Fluent: {np.sum(y_full == 0)} ({np.sum(y_full == 0)/len(y_full)*100:.1f}%)")
    print(f"  Stutter: {np.sum(y_full == 1)} ({np.sum(y_full == 1)/len(y_full)*100:.1f}%)")
    print("\nNext step: Run hybrid.py to train models")
    print("  Command: py hybrid.py")
    print("="*60)

if __name__ == "__main__":
    main()
