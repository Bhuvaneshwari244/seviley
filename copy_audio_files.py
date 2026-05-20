import os
import shutil
from pathlib import Path

# Source: Your existing audio files
SOURCE_DIR = r"D:\sevilay\data\clips\stuttering-clips\clips"

# Destination: Project's processed_clips folder
DEST_DIR = os.path.join(os.path.dirname(__file__), "processed_clips")

def copy_audio_files():
    print("="*60)
    print(" COPYING AUDIO FILES TO PROJECT ")
    print("="*60)
    
    # Check if source exists
    if not os.path.exists(SOURCE_DIR):
        print(f"[ERROR] Source folder not found: {SOURCE_DIR}")
        print("Please check the path to your audio files.")
        return
    
    # Create destination folder if it doesn't exist
    os.makedirs(DEST_DIR, exist_ok=True)
    
    # Get all WAV files from source
    source_files = list(Path(SOURCE_DIR).glob("*.wav"))
    
    if len(source_files) == 0:
        print(f"[WARNING] No WAV files found in: {SOURCE_DIR}")
        return
    
    print(f"Found {len(source_files)} WAV files in source folder")
    print(f"Copying to: {DEST_DIR}")
    print()
    
    copied = 0
    skipped = 0
    
    for source_file in source_files:
        dest_file = os.path.join(DEST_DIR, source_file.name)
        
        # Skip if file already exists
        if os.path.exists(dest_file):
            skipped += 1
            continue
        
        try:
            shutil.copy2(source_file, dest_file)
            copied += 1
            
            # Progress indicator
            if copied % 100 == 0:
                print(f"Copied {copied} files...")
        except Exception as e:
            print(f"[ERROR] Failed to copy {source_file.name}: {e}")
    
    print()
    print("="*60)
    print(" COPY COMPLETE ")
    print("="*60)
    print(f"Total files found  : {len(source_files)}")
    print(f"Files copied       : {copied}")
    print(f"Files skipped      : {skipped}")
    print(f"Destination folder : {DEST_DIR}")
    print("="*60)

if __name__ == "__main__":
    copy_audio_files()
