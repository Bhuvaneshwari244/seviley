"""
Fixed Audio Downloader for SEP-28k Dataset
"""

import os
import pathlib
import subprocess
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Download raw audio files for SEP-28k or FluencyBank')
parser.add_argument('--episodes', type=str, required=True)
parser.add_argument('--wavs', type=str, default="raw_audio")
args = parser.parse_args()

# Load episode data using pandas (no header)
df = pd.read_csv(args.episodes, header=None, names=['Title', 'Episode', 'Url', 'Show', 'EpId'])
wav_dir = args.wavs

audio_types = [".mp3", ".m4a", ".mp4"]

print(f"Total episodes to download: {len(df)}")

for idx, row in df.iterrows():
    show_abrev = str(row['Show']).strip()
    ep_idx = str(row['EpId']).strip()
    episode_url = str(row['Url']).strip()
    
    # Check file extension
    ext = ''
    for e in audio_types:
        if e in episode_url:
            ext = e
            break
    
    if not ext:
        print(f"Skipping {show_abrev}/{ep_idx} - unknown format")
        continue
    
    # Ensure the base folder exists
    episode_dir = pathlib.Path(f"{wav_dir}/{show_abrev}")
    os.makedirs(episode_dir, exist_ok=True)
    
    # Get file paths
    audio_path_orig = pathlib.Path(f"{episode_dir}/{ep_idx}{ext}")
    wav_path = pathlib.Path(f"{episode_dir}/{ep_idx}.wav")
    
    # Check if already downloaded
    if os.path.exists(wav_path):
        print(f"[{idx+1}/{len(df)}] Skipping {show_abrev}/{ep_idx} - already exists")
        continue
    
    print(f"[{idx+1}/{len(df)}] Processing {show_abrev}/{ep_idx}")
    
    # Download raw audio
    if not os.path.exists(audio_path_orig):
        try:
            subprocess.run([
                "wget.exe", "-O", str(audio_path_orig), episode_url
            ], check=True, timeout=300)
        except Exception as e:
            print(f"  Error downloading: {e}")
            continue
    
    # Convert to 16khz mono wav
    try:
        subprocess.run([
            "ffmpeg", "-i", str(audio_path_orig),
            "-ac", "1", "-ar", "16000",
            str(wav_path), "-y"
        ], check=True, capture_output=True, timeout=300)
        print(f"  ✓ Converted to WAV")
    except Exception as e:
        print(f"  Error converting: {e}")
        continue
    
    # Remove original file
    try:
        if os.path.exists(audio_path_orig):
            os.remove(audio_path_orig)
    except:
        pass

print("\nDownload complete!")
