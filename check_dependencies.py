"""
Check if all required libraries are installed for the stuttering detection project.
"""

import sys

def check_library(name, import_name=None):
    """Try to import a library and report status."""
    if import_name is None:
        import_name = name
    
    try:
        __import__(import_name)
        print(f"✓ {name:20s} - Installed")
        return True
    except ImportError:
        print(f"✗ {name:20s} - NOT INSTALLED")
        return False

def main():
    print("="*60)
    print(" CHECKING DEPENDENCIES ")
    print("="*60)
    print()
    
    libraries = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("scikit-learn", "sklearn"),
        ("xgboost", "xgboost"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("tqdm", "tqdm"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("evaluate", "evaluate"),
        ("accelerate", "accelerate"),
    ]
    
    installed = []
    missing = []
    
    for name, import_name in libraries:
        if check_library(name, import_name):
            installed.append(name)
        else:
            missing.append(name)
    
    print()
    print("="*60)
    print(" SUMMARY ")
    print("="*60)
    print(f"Installed: {len(installed)}/{len(libraries)}")
    print(f"Missing: {len(missing)}/{len(libraries)}")
    
    if missing:
        print()
        print("To install missing libraries, run:")
        print()
        print(f"py -m pip install {' '.join(missing)}")
        print()
        print("Or install all at once:")
        print("py -m pip install -r requirements.txt")
    else:
        print()
        print("✓ All dependencies are installed!")
        print("You can now run the training scripts.")
    
    print("="*60)

if __name__ == "__main__":
    main()
