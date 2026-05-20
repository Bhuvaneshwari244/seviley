"""
Configuration File for Stuttering Detection Project
----------------------------------------------------
Centralized configuration to avoid hardcoded paths.
"""

import os

# =====================================================
# PROJECT PATHS
# =====================================================
# Base directory (automatically detected)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
AUDIO_DIR = os.path.join(BASE_DIR, "processed_clips")
RAW_AUDIO_DIR = os.path.join(BASE_DIR, "raw_audio")

# CSV files
SEP28K_LABELS = os.path.join(BASE_DIR, "SEP-28k_labels.csv")
FLUENCYBANK_LABELS = os.path.join(BASE_DIR, "FluencyBank_labels.csv")
PROCESSED_LABELS = os.path.join(BASE_DIR, "processed_labels.csv")

# Split files
TRAIN_SPLIT = os.path.join(BASE_DIR, "train_split.csv")
VAL_SPLIT = os.path.join(BASE_DIR, "val_split.csv")
TEST_SPLIT = os.path.join(BASE_DIR, "test_split.csv")

# Feature files
EMBEDDING_FEATURES = os.path.join(BASE_DIR, "embedding_features.npy")
EMBEDDING_LABELS = os.path.join(BASE_DIR, "embedding_labels.npy")
SELECTED_FEATURES = os.path.join(BASE_DIR, "selected_features.npy")
SELECTED_LABELS = os.path.join(BASE_DIR, "selected_labels.npy")

# Model directories
HUBERT_MODEL_DIR = os.path.join(BASE_DIR, "best_hubert_stuttering_model")
ENSEMBLE_OUTPUT_DIR = os.path.join(BASE_DIR, "ensemble_outputs")

# =====================================================
# MODEL PARAMETERS
# =====================================================
# HuBERT
HUBERT_BASE_MODEL = "facebook/hubert-base-ls960"
MAX_AUDIO_LENGTH = 16000 * 3  # 3 seconds at 16kHz
SAMPLE_RATE = 16000

# Feature Selection
K_FEATURES = 256

# Training
BATCH_SIZE = 8
EPOCHS = 5
RANDOM_SEED = 42

# Data Split
TEST_SIZE = 0.20
VAL_SIZE = 0.15
N_FOLDS = 5

# =====================================================
# DEVICE CONFIGURATION
# =====================================================
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    os.makedirs(directory, exist_ok=True)
    return directory

def get_audio_filename(show, ep_id, clip_id):
    """
    Generate standardized audio filename.
    Handles FluencyBank EpId padding.
    """
    show = str(show).strip()
    ep_id = str(ep_id).strip()
    clip_id = str(clip_id).strip()
    
    # FluencyBank requires 3-digit EpId
    if "fluencybank" in show.lower():
        ep_id = ep_id.zfill(3)
    
    return f"{show}_{ep_id}_{clip_id}.wav"

def get_audio_path(show, ep_id, clip_id, audio_dir=None):
    """Get full path to audio file"""
    if audio_dir is None:
        audio_dir = AUDIO_DIR
    filename = get_audio_filename(show, ep_id, clip_id)
    return os.path.join(audio_dir, filename)

# =====================================================
# PRINT CONFIGURATION
# =====================================================
if __name__ == "__main__":
    print("="*60)
    print(" PROJECT CONFIGURATION")
    print("="*60)
    print(f"\nBase Directory: {BASE_DIR}")
    print(f"Audio Directory: {AUDIO_DIR}")
    print(f"Device: {DEVICE}")
    print(f"\nModel Parameters:")
    print(f"  HuBERT Model: {HUBERT_BASE_MODEL}")
    print(f"  Sample Rate: {SAMPLE_RATE} Hz")
    print(f"  Max Length: {MAX_AUDIO_LENGTH / SAMPLE_RATE:.1f} seconds")
    print(f"  K Features: {K_FEATURES}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Random Seed: {RANDOM_SEED}")
    print("="*60)
