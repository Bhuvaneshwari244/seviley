"""
Annotator Agreement Analysis for SEP-28k + FluencyBank
-------------------------------------------------------
Analyzes inter-annotator agreement and its impact on model performance.
This provides evidence for your uncertainty-aware training contribution.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
import seaborn as sns

def analyze_annotator_agreement():
    """
    Computes inter-annotator agreement metrics and visualizes disagreement patterns.
    """
    df = pd.read_csv("processed_labels.csv")
    
    stutter_cols = ['Prolongation', 'Block', 'SoundRep', 'WordRep']
    
    # 1. Calculate agreement distribution
    print("="*60)
    print(" INTER-ANNOTATOR AGREEMENT ANALYSIS")
    print("="*60)
    
    # Stutter ratio distribution
    ratios = df['stutter_ratio'].values
    
    # Categorize agreement levels
    unanimous_stutter = (ratios == 1.0).sum()
    unanimous_fluent = (ratios == 0.0).sum()
    majority_stutter = ((ratios >= 0.5) & (ratios < 1.0)).sum()
    majority_fluent = ((ratios > 0.0) & (ratios < 0.5)).sum()
    
    print(f"\nAgreement Distribution:")
    print(f"  Unanimous Stutter (3/3): {unanimous_stutter:>6} ({unanimous_stutter/len(df)*100:.1f}%)")
    print(f"  Majority Stutter  (2/3): {majority_stutter:>6} ({majority_stutter/len(df)*100:.1f}%)")
    print(f"  Majority Fluent   (1/3): {majority_fluent:>6} ({majority_fluent/len(df)*100:.1f}%)")
    print(f"  Unanimous Fluent  (0/3): {unanimous_fluent:>6} ({unanimous_fluent/len(df)*100:.1f}%)")
    
    # 2. Stuttering type distribution
    print(f"\n\nStuttering Event Type Distribution:")
    for col in stutter_cols:
        if col in df.columns:
            count = (df[col] > 0).sum()
            print(f"  {col:<20}: {count:>6} clips ({count/len(df)*100:.1f}%)")
    
    # 3. Calculate annotation confidence
    df['annotation_confidence'] = 1 - 2 * np.abs(df['stutter_ratio'] - 0.5)
    
    high_conf = (df['annotation_confidence'] >= 0.8).sum()
    medium_conf = ((df['annotation_confidence'] >= 0.4) & (df['annotation_confidence'] < 0.8)).sum()
    low_conf = (df['annotation_confidence'] < 0.4).sum()
    
    print(f"\n\nAnnotation Confidence Levels:")
    print(f"  High   (≥0.8): {high_conf:>6} ({high_conf/len(df)*100:.1f}%)")
    print(f"  Medium (0.4-0.8): {medium_conf:>6} ({medium_conf/len(df)*100:.1f}%)")
    print(f"  Low    (<0.4): {low_conf:>6} ({low_conf/len(df)*100:.1f}%)")
    
    # 4. Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Stutter ratio distribution
    axes[0, 0].hist(df['stutter_ratio'], bins=20, color='steelblue', edgecolor='black')
    axes[0, 0].set_xlabel('Stutter Ratio (Annotator Agreement)')
    axes[0, 0].set_ylabel('Number of Clips')
    axes[0, 0].set_title('Distribution of Annotator Agreement')
    axes[0, 0].axvline(0.5, color='red', linestyle='--', label='Decision Threshold')
    axes[0, 0].legend()
    
    # Plot 2: Confidence distribution
    axes[0, 1].hist(df['annotation_confidence'], bins=20, color='coral', edgecolor='black')
    axes[0, 1].set_xlabel('Annotation Confidence')
    axes[0, 1].set_ylabel('Number of Clips')
    axes[0, 1].set_title('Annotation Confidence Distribution')
    
    # Plot 3: Event type co-occurrence
    if all(col in df.columns for col in stutter_cols):
        event_counts = df[stutter_cols].sum().sort_values(ascending=False)
        axes[1, 0].bar(range(len(event_counts)), event_counts.values, color='teal')
        axes[1, 0].set_xticks(range(len(event_counts)))
        axes[1, 0].set_xticklabels(event_counts.index, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Total Annotator Votes')
        axes[1, 0].set_title('Stuttering Event Type Frequency')
    
    # Plot 4: Confidence vs Label
    stutter_conf = df[df['is_stutter'] == 1]['annotation_confidence']
    fluent_conf = df[df['is_stutter'] == 0]['annotation_confidence']
    
    axes[1, 1].hist([fluent_conf, stutter_conf], bins=15, 
                    label=['Fluent', 'Stutter'], color=['lightgreen', 'salmon'], 
                    edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Annotation Confidence')
    axes[1, 1].set_ylabel('Number of Clips')
    axes[1, 1].set_title('Confidence Distribution by Label')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('annotator_agreement_analysis.png', dpi=150)
    print(f"\n\n[✓] Visualization saved: annotator_agreement_analysis.png")
    
    # 5. Save confidence scores for weighted training
    df[['Show', 'EpId', 'ClipId', 'is_stutter', 'stutter_ratio', 
        'annotation_confidence']].to_csv('labels_with_confidence.csv', index=False)
    print(f"[✓] Confidence scores saved: labels_with_confidence.csv")
    
    return df

if __name__ == "__main__":
    analyze_annotator_agreement()
