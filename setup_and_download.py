"""
Automated Audio Download Setup for SEP-28k Dataset
---------------------------------------------------
This script helps you download all audio clips automatically.
"""

import os
import subprocess
import sys

def check_tool(tool_name, install_cmd):
    """Check if a tool is installed"""
    try:
        result = subprocess.run([tool_name, "--version"], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        return True
    except:
        return False

def install_tools():
    """Guide user to install required tools"""
    print("="*60)
    print(" CHECKING REQUIRED TOOLS")
    print("="*60)
    
    wget_installed = check_tool("wget", "winget install GnuWin32.Wget")
    ffmpeg_installed = check_tool("ffmpeg", "winget install Gyan.FFmpeg")
    
    if wget_installed:
        print("✓ wget is installed")
    else:
        print("✗ wget is NOT installed")
        print("\nTo install wget, run this command in PowerShell (as Administrator):")
        print("  winget install GnuWin32.Wget")
        print("\nOr download from: https://eternallybored.org/misc/wget/")
    
    if ffmpeg_installed:
        print("✓ ffmpeg is installed")
    else:
        print("✗ ffmpeg is NOT installed")
        print("\nTo install ffmpeg, run this command in PowerShell (as Administrator):")
        print("  winget install Gyan.FFmpeg")
        print("\nOr download from: https://www.gyan.dev/ffmpeg/builds/")
    
    if not (wget_installed and ffmpeg_installed):
        print("\n" + "="*60)
        print(" INSTALLATION REQUIRED")
        print("="*60)
        print("\nAfter installing, restart your terminal and run this script again.")
        return False
    
    return True

def download_audio():
    """Download audio files using the provided scripts"""
    print("\n" + "="*60)
    print(" STARTING AUDIO DOWNLOAD")
    print("="*60)
    
    # Create directories
    os.makedirs("raw_audio", exist_ok=True)
    os.makedirs("processed_clips", exist_ok=True)
    
    print("\nThis will download:")
    print("  - SEP-28k: ~28,000 clips (~32 GB raw, ~2.6 GB processed)")
    print("  - FluencyBank: ~4,000 clips (~6 GB raw, ~400 MB processed)")
    print("\nEstimated time: 4-8 hours (depends on internet speed)")
    print("Total disk space needed: ~40 GB")
    
    response = input("\nDo you want to continue? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("Download cancelled.")
        return False
    
    # Download SEP-28k
    print("\n" + "-"*60)
    print(" STEP 1: Downloading SEP-28k audio...")
    print("-"*60)
    
    try:
        subprocess.run([
            sys.executable, "download_audio.py",
            "--episodes", "SEP-28k_episodes.csv",
            "--wavs", "raw_audio"
        ], check=True)
        print("✓ SEP-28k audio downloaded")
    except Exception as e:
        print(f"✗ Error downloading SEP-28k: {e}")
        return False
    
    # Extract SEP-28k clips
    print("\n" + "-"*60)
    print(" STEP 2: Extracting SEP-28k clips...")
    print("-"*60)
    
    try:
        subprocess.run([
            sys.executable, "extract_clips.py",
            "--labels", "SEP-28k_labels.csv",
            "--wavs", "raw_audio",
            "--clips", "processed_clips",
            "--progress"
        ], check=True)
        print("✓ SEP-28k clips extracted")
    except Exception as e:
        print(f"✗ Error extracting SEP-28k: {e}")
        return False
    
    # Download FluencyBank
    print("\n" + "-"*60)
    print(" STEP 3: Downloading FluencyBank audio...")
    print("-"*60)
    
    try:
        subprocess.run([
            sys.executable, "download_audio.py",
            "--episodes", "fluencybank_episodes.csv",
            "--wavs", "raw_audio"
        ], check=True)
        print("✓ FluencyBank audio downloaded")
    except Exception as e:
        print(f"✗ Error downloading FluencyBank: {e}")
        return False
    
    # Extract FluencyBank clips
    print("\n" + "-"*60)
    print(" STEP 4: Extracting FluencyBank clips...")
    print("-"*60)
    
    try:
        subprocess.run([
            sys.executable, "extract_clips.py",
            "--labels", "fluencybank_labels.csv",
            "--wavs", "raw_audio",
            "--clips", "processed_clips",
            "--progress"
        ], check=True)
        print("✓ FluencyBank clips extracted")
    except Exception as e:
        print(f"✗ Error extracting FluencyBank: {e}")
        return False
    
    return True

def verify_download():
    """Verify that clips were downloaded successfully"""
    print("\n" + "="*60)
    print(" VERIFYING DOWNLOAD")
    print("="*60)
    
    if not os.path.exists("processed_clips"):
        print("✗ processed_clips folder not found")
        return False
    
    # Count clips
    clip_count = 0
    for root, dirs, files in os.walk("processed_clips"):
        clip_count += len([f for f in files if f.endswith('.wav')])
    
    print(f"\nTotal clips downloaded: {clip_count:,}")
    
    if clip_count > 30000:
        print("✓ Download appears complete!")
        return True
    elif clip_count > 0:
        print("⚠ Partial download - some clips may be missing")
        return True
    else:
        print("✗ No clips found")
        return False

def main():
    print("="*60)
    print(" SEP-28K AUDIO DOWNLOAD AUTOMATION")
    print("="*60)
    
    # Step 1: Check tools
    if not install_tools():
        print("\n⚠ Please install required tools first, then run this script again.")
        return
    
    # Step 2: Download audio
    if not download_audio():
        print("\n✗ Download failed. Check errors above.")
        return
    
    # Step 3: Verify
    if verify_download():
        print("\n" + "="*60)
        print(" ✓ DOWNLOAD COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("  1. py clean_data.py              # Verify audio quality")
        print("  2. py split_data.py              # Create train/val/test splits")
        print("  3. py train_transfer_hubert.py   # Train model")
    else:
        print("\n✗ Verification failed")

if __name__ == "__main__":
    main()
