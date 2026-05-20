"""
Master script to run complete training pipeline for stuttering detection.
This will take 12-15 hours to complete (mostly HuBERT training).
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_script(script_name, description, estimated_time):
    """Run a Python script and track its execution."""
    print("\n" + "="*70)
    print(f" {description}")
    print("="*70)
    print(f"Script: {script_name}")
    print(f"Estimated time: {estimated_time}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        
        print("\n" + "="*70)
        print(f"✓ {description} - COMPLETED")
        print(f"Actual time: {hours}h {minutes}m")
        print("="*70)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print("\n" + "="*70)
        print(f"✗ {description} - FAILED")
        print(f"Error: {e}")
        print("="*70)
        return False
    except KeyboardInterrupt:
        print("\n" + "="*70)
        print("Training interrupted by user")
        print("="*70)
        return False

def main():
    print("="*70)
    print(" STUTTERING DETECTION - COMPLETE TRAINING PIPELINE ")
    print("="*70)
    print()
    print("This will run all training scripts in sequence:")
    print("1. Data verification")
    print("2. HuBERT model training (8-12 hours)")
    print("3. Feature extraction (30-60 min)")
    print("4. Feature selection (5-10 min)")
    print("5. Hybrid ensemble training (1-2 hours)")
    print("6. Baseline comparison (5 min)")
    print("7. Results report generation (5 min)")
    print()
    print("Total estimated time: 12-15 hours")
    print()
    
    response = input("Do you want to continue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Training cancelled.")
        return
    
    start_time = time.time()
    
    # Pipeline steps
    steps = [
        ("verify_audio_files.py", "Step 1: Data Verification", "1 minute"),
        ("train_transfer_hubert.py", "Step 2: HuBERT Training", "8-12 hours"),
        ("Gap_feature.py", "Step 3: Feature Extraction", "30-60 minutes"),
        ("SelectKbest_feature_select.py", "Step 4: Feature Selection", "5-10 minutes"),
        ("hybrid.py", "Step 5: Hybrid Ensemble Training", "1-2 hours"),
        ("baseline_comparison.py", "Step 6: Baseline Comparison", "5 minutes"),
        ("generate_thesis_report.py", "Step 7: Results Report", "5 minutes"),
    ]
    
    completed = []
    failed = []
    
    for script, description, est_time in steps:
        if not os.path.exists(script):
            print(f"\n✗ Script not found: {script}")
            failed.append(script)
            continue
        
        success = run_script(script, description, est_time)
        
        if success:
            completed.append(script)
        else:
            failed.append(script)
            print(f"\nTraining stopped due to error in {script}")
            break
    
    # Final summary
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    
    print("\n" + "="*70)
    print(" TRAINING PIPELINE SUMMARY ")
    print("="*70)
    print(f"Completed: {len(completed)}/{len(steps)} steps")
    print(f"Failed: {len(failed)} steps")
    print(f"Total time: {hours}h {minutes}m")
    print()
    
    if completed:
        print("Completed steps:")
        for script in completed:
            print(f"  ✓ {script}")
    
    if failed:
        print("\nFailed steps:")
        for script in failed:
            print(f"  ✗ {script}")
    
    print("="*70)
    
    if len(completed) == len(steps):
        print("\n🎉 ALL TRAINING COMPLETE! 🎉")
        print("\nYour project is now 100% complete!")
        print("Check the following files for results:")
        print("  - best_hubert_stuttering_model/ (trained model)")
        print("  - THESIS_RESULTS.txt (complete results)")
        print("  - Various .png files (visualizations)")
        print("\nYou can now write your thesis results chapter!")
    else:
        print("\nTraining incomplete. Please check errors above.")

if __name__ == "__main__":
    main()
