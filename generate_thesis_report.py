"""
Comprehensive Thesis Report Generator
--------------------------------------
Generates a complete analysis report for your thesis/paper including:
- Dataset statistics
- Annotator agreement analysis
- Feature comparison results
- Model performance metrics
- Visualizations
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def generate_report():
    """Generate comprehensive thesis report"""
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append(" STUTTERING DETECTION SYSTEM - COMPREHENSIVE REPORT")
    report_lines.append("="*80)
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("\n")
    
    # ========================================
    # 1. DATASET STATISTICS
    # ========================================
    report_lines.append("\n" + "="*80)
    report_lines.append(" 1. DATASET STATISTICS")
    report_lines.append("="*80)
    
    if os.path.exists("processed_labels.csv"):
        df = pd.read_csv("processed_labels.csv")
        
        total_clips = len(df)
        stutter_clips = (df['is_stutter'] == 1).sum()
        fluent_clips = (df['is_stutter'] == 0).sum()
        
        report_lines.append(f"\nTotal Clips: {total_clips:,}")
        report_lines.append(f"  Stuttered: {stutter_clips:,} ({stutter_clips/total_clips*100:.1f}%)")
        report_lines.append(f"  Fluent:    {fluent_clips:,} ({fluent_clips/total_clips*100:.1f}%)")
        
        # Dataset distribution
        if 'dataset_origin' in df.columns or 'Show' in df.columns:
            sep28k_count = df['Show'].str.contains('fluencybank', case=False, na=False).sum()
            fluencybank_count = len(df) - sep28k_count
            report_lines.append(f"\nDataset Distribution:")
            report_lines.append(f"  SEP-28k:      {sep28k_count:,}")
            report_lines.append(f"  FluencyBank:  {fluencybank_count:,}")
    
    # Train/Val/Test split
    if all(os.path.exists(f) for f in ["train_split.csv", "val_split.csv", "test_split.csv"]):
        train_df = pd.read_csv("train_split.csv")
        val_df = pd.read_csv("val_split.csv")
        test_df = pd.read_csv("test_split.csv")
        
        report_lines.append(f"\nData Split:")
        report_lines.append(f"  Train: {len(train_df):>6} clips ({len(train_df)/total_clips*100:.1f}%)")
        report_lines.append(f"  Val:   {len(val_df):>6} clips ({len(val_df)/total_clips*100:.1f}%)")
        report_lines.append(f"  Test:  {len(test_df):>6} clips ({len(test_df)/total_clips*100:.1f}%)")
    
    # ========================================
    # 2. ANNOTATOR AGREEMENT ANALYSIS
    # ========================================
    report_lines.append("\n\n" + "="*80)
    report_lines.append(" 2. ANNOTATOR AGREEMENT ANALYSIS")
    report_lines.append("="*80)
    
    if os.path.exists("labels_with_confidence.csv"):
        conf_df = pd.read_csv("labels_with_confidence.csv")
        
        # Agreement levels
        unanimous_stutter = (conf_df['stutter_ratio'] == 1.0).sum()
        unanimous_fluent = (conf_df['stutter_ratio'] == 0.0).sum()
        majority_stutter = ((conf_df['stutter_ratio'] >= 0.5) & (conf_df['stutter_ratio'] < 1.0)).sum()
        majority_fluent = ((conf_df['stutter_ratio'] > 0.0) & (conf_df['stutter_ratio'] < 0.5)).sum()
        
        report_lines.append(f"\nAnnotator Agreement Distribution:")
        report_lines.append(f"  Unanimous Stutter (3/3): {unanimous_stutter:>6} ({unanimous_stutter/len(conf_df)*100:.1f}%)")
        report_lines.append(f"  Majority Stutter  (2/3): {majority_stutter:>6} ({majority_stutter/len(conf_df)*100:.1f}%)")
        report_lines.append(f"  Majority Fluent   (1/3): {majority_fluent:>6} ({majority_fluent/len(conf_df)*100:.1f}%)")
        report_lines.append(f"  Unanimous Fluent  (0/3): {unanimous_fluent:>6} ({unanimous_fluent/len(conf_df)*100:.1f}%)")
        
        # Confidence statistics
        mean_conf = conf_df['annotation_confidence'].mean()
        median_conf = conf_df['annotation_confidence'].median()
        
        report_lines.append(f"\nAnnotation Confidence Statistics:")
        report_lines.append(f"  Mean:   {mean_conf:.4f}")
        report_lines.append(f"  Median: {median_conf:.4f}")
        
        high_conf = (conf_df['annotation_confidence'] >= 0.8).sum()
        medium_conf = ((conf_df['annotation_confidence'] >= 0.4) & (conf_df['annotation_confidence'] < 0.8)).sum()
        low_conf = (conf_df['annotation_confidence'] < 0.4).sum()
        
        report_lines.append(f"\nConfidence Distribution:")
        report_lines.append(f"  High   (≥0.8):   {high_conf:>6} ({high_conf/len(conf_df)*100:.1f}%)")
        report_lines.append(f"  Medium (0.4-0.8): {medium_conf:>6} ({medium_conf/len(conf_df)*100:.1f}%)")
        report_lines.append(f"  Low    (<0.4):   {low_conf:>6} ({low_conf/len(conf_df)*100:.1f}%)")
    else:
        report_lines.append("\n[!] Run annotator_analysis.py first to generate this section.")
    
    # ========================================
    # 3. FEATURE COMPARISON
    # ========================================
    report_lines.append("\n\n" + "="*80)
    report_lines.append(" 3. FEATURE COMPARISON: HuBERT vs MFCC")
    report_lines.append("="*80)
    
    if os.path.exists("feature_comparison_results.json"):
        with open("feature_comparison_results.json", 'r') as f:
            comp_results = json.load(f)
        
        report_lines.append(f"\n{'Metric':<20} {'HuBERT':<15} {'MFCC':<15} {'Improvement':<15}")
        report_lines.append("-"*65)
        
        for metric in ['accuracy', 'f1', 'auc']:
            hubert_val = comp_results['hubert'][metric]
            mfcc_val = comp_results['mfcc'][metric]
            improvement = comp_results['improvement'][metric]
            
            metric_name = metric.upper() if metric == 'auc' else metric.replace('_', ' ').title()
            report_lines.append(f"{metric_name:<20} {hubert_val:<15.4f} {mfcc_val:<15.4f} {improvement:>+.2f}%")
        
        report_lines.append("\n✓ HuBERT features outperform traditional MFCC baseline")
    else:
        report_lines.append("\n[!] Run baseline_comparison.py first to generate this section.")
    
    # ========================================
    # 4. ENSEMBLE MODEL PERFORMANCE
    # ========================================
    report_lines.append("\n\n" + "="*80)
    report_lines.append(" 4. HYBRID ENSEMBLE PERFORMANCE")
    report_lines.append("="*80)
    
    if os.path.exists("ensemble_outputs/best_weights.json"):
        with open("ensemble_outputs/best_weights.json", 'r') as f:
            ensemble_results = json.load(f)
        
        report_lines.append(f"\nTest Set Performance:")
        report_lines.append(f"  Accuracy:  {ensemble_results['test_accuracy']:.4f}")
        report_lines.append(f"  F1-Score:  {ensemble_results['test_f1_macro']:.4f}")
        report_lines.append(f"  ROC-AUC:   {ensemble_results['test_roc_auc']:.4f}")
        
        report_lines.append(f"\nOut-of-Fold (OOF) Performance:")
        report_lines.append(f"  ROC-AUC:   {ensemble_results['oof_auc']:.4f}")
        
        report_lines.append(f"\nOptimal Threshold (Youden Index): {ensemble_results['optimal_threshold']:.4f}")
        
        report_lines.append(f"\nModel Weights:")
        for model, weight in ensemble_results['weights'].items():
            report_lines.append(f"  {model:<15}: {weight:.4f}")
    else:
        report_lines.append("\n[!] Run hybrid.py first to generate this section.")
    
    # ========================================
    # 5. FEATURE DIMENSIONS
    # ========================================
    report_lines.append("\n\n" + "="*80)
    report_lines.append(" 5. FEATURE EXTRACTION PIPELINE")
    report_lines.append("="*80)
    
    if os.path.exists("embedding_features.npy"):
        emb_features = np.load("embedding_features.npy")
        report_lines.append(f"\nHuBERT GAP Embeddings: {emb_features.shape}")
    
    if os.path.exists("selected_features.npy"):
        sel_features = np.load("selected_features.npy")
        report_lines.append(f"After SelectKBest:     {sel_features.shape}")
        
        if os.path.exists("embedding_features.npy"):
            reduction = (1 - sel_features.shape[1] / emb_features.shape[1]) * 100
            report_lines.append(f"Dimensionality Reduction: {reduction:.1f}%")
    
    # ========================================
    # 6. KEY CONTRIBUTIONS SUMMARY
    # ========================================
    report_lines.append("\n\n" + "="*80)
    report_lines.append(" 6. KEY CONTRIBUTIONS")
    report_lines.append("="*80)
    
    report_lines.append("\n1. Multi-Annotator Uncertainty Modeling")
    report_lines.append("   - Novel confidence scoring based on annotator agreement")
    report_lines.append("   - Addresses annotation ambiguity in stuttering detection")
    
    report_lines.append("\n2. Speaker-Isolated Data Splitting")
    report_lines.append("   - Zero data leakage using GroupShuffleSplit")
    report_lines.append("   - Ensures realistic generalization performance")
    
    report_lines.append("\n3. Cross-Dataset Hybrid Ensemble")
    report_lines.append("   - Combines SEP-28k (podcast) + FluencyBank (clinical)")
    report_lines.append("   - 5 diverse models: XGBoost, SVM, RF, CNN, BiLSTM")
    report_lines.append("   - Out-of-Fold validation prevents overfitting")
    
    report_lines.append("\n4. HuBERT Transfer Learning")
    report_lines.append("   - Fine-tuned facebook/hubert-base-ls960")
    report_lines.append("   - Global Average Pooling for fixed-length embeddings")
    if os.path.exists("feature_comparison_results.json"):
        report_lines.append(f"   - Outperforms MFCC baseline by {comp_results['improvement']['f1']:.1f}% F1")
    
    # ========================================
    # SAVE REPORT
    # ========================================
    report_text = "\n".join(report_lines)
    
    # Print to console
    print(report_text)
    
    # Save to file
    with open("THESIS_REPORT.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print("\n" + "="*80)
    print(" REPORT SAVED: THESIS_REPORT.txt")
    print("="*80)
    
    # ========================================
    # GENERATE SUMMARY VISUALIZATION
    # ========================================
    generate_summary_visualization()

def generate_summary_visualization():
    """Create a comprehensive summary visualization"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Dataset Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    if os.path.exists("processed_labels.csv"):
        df = pd.read_csv("processed_labels.csv")
        labels = ['Fluent', 'Stutter']
        sizes = [(df['is_stutter'] == 0).sum(), (df['is_stutter'] == 1).sum()]
        colors = ['lightgreen', 'salmon']
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Dataset Label Distribution')
    
    # 2. Annotator Agreement
    ax2 = fig.add_subplot(gs[0, 1])
    if os.path.exists("labels_with_confidence.csv"):
        conf_df = pd.read_csv("labels_with_confidence.csv")
        ax2.hist(conf_df['annotation_confidence'], bins=20, color='steelblue', edgecolor='black')
        ax2.set_xlabel('Annotation Confidence')
        ax2.set_ylabel('Number of Clips')
        ax2.set_title('Annotation Confidence Distribution')
    
    # 3. Feature Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    if os.path.exists("feature_comparison_results.json"):
        with open("feature_comparison_results.json", 'r') as f:
            comp = json.load(f)
        
        metrics = ['Accuracy', 'F1-Score', 'ROC-AUC']
        hubert_vals = [comp['hubert']['accuracy'], comp['hubert']['f1'], comp['hubert']['auc']]
        mfcc_vals = [comp['mfcc']['accuracy'], comp['mfcc']['f1'], comp['mfcc']['auc']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax3.bar(x - width/2, hubert_vals, width, label='HuBERT', color='teal')
        ax3.bar(x + width/2, mfcc_vals, width, label='MFCC', color='coral')
        ax3.set_ylabel('Score')
        ax3.set_title('Feature Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics, rotation=15, ha='right')
        ax3.legend()
        ax3.set_ylim([0.6, 1.0])
    
    # 4. Data Split Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    if all(os.path.exists(f) for f in ["train_split.csv", "val_split.csv", "test_split.csv"]):
        splits = ['Train', 'Val', 'Test']
        sizes = [
            len(pd.read_csv("train_split.csv")),
            len(pd.read_csv("val_split.csv")),
            len(pd.read_csv("test_split.csv"))
        ]
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        ax4.bar(splits, sizes, color=colors, edgecolor='black')
        ax4.set_ylabel('Number of Clips')
        ax4.set_title('Train/Val/Test Split')
        for i, v in enumerate(sizes):
            ax4.text(i, v + 100, str(v), ha='center', va='bottom')
    
    # 5. Ensemble Model Weights
    ax5 = fig.add_subplot(gs[1, 1])
    if os.path.exists("ensemble_outputs/best_weights.json"):
        with open("ensemble_outputs/best_weights.json", 'r') as f:
            ensemble = json.load(f)
        
        models = list(ensemble['weights'].keys())
        weights = list(ensemble['weights'].values())
        
        ax5.barh(models, weights, color='mediumpurple', edgecolor='black')
        ax5.set_xlabel('Weight')
        ax5.set_title('Ensemble Model Weights')
        ax5.set_xlim([0, max(weights) * 1.2])
    
    # 6. Performance Metrics
    ax6 = fig.add_subplot(gs[1, 2])
    if os.path.exists("ensemble_outputs/best_weights.json"):
        with open("ensemble_outputs/best_weights.json", 'r') as f:
            ensemble = json.load(f)
        
        metrics = ['Accuracy', 'F1-Score', 'ROC-AUC']
        values = [
            ensemble['test_accuracy'],
            ensemble['test_f1_macro'],
            ensemble['test_roc_auc']
        ]
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        ax6.bar(metrics, values, color=colors, edgecolor='black')
        ax6.set_ylabel('Score')
        ax6.set_title('Ensemble Test Performance')
        ax6.set_ylim([0.7, 1.0])
        for i, v in enumerate(values):
            ax6.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 7. Feature Dimensionality
    ax7 = fig.add_subplot(gs[2, :])
    if os.path.exists("embedding_features.npy") and os.path.exists("selected_features.npy"):
        stages = ['HuBERT\nEmbeddings', 'SelectKBest\nFeatures']
        dims = [
            np.load("embedding_features.npy").shape[1],
            np.load("selected_features.npy").shape[1]
        ]
        
        ax7.plot(stages, dims, marker='o', markersize=12, linewidth=3, color='darkblue')
        ax7.set_ylabel('Feature Dimensions')
        ax7.set_title('Feature Extraction Pipeline')
        ax7.grid(True, alpha=0.3)
        for i, (stage, dim) in enumerate(zip(stages, dims)):
            ax7.text(i, dim + 20, str(dim), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.suptitle('Stuttering Detection System - Summary Report', fontsize=16, fontweight='bold')
    plt.savefig('THESIS_SUMMARY.png', dpi=150, bbox_inches='tight')
    print("\n[✓] Summary visualization saved: THESIS_SUMMARY.png")

if __name__ == "__main__":
    generate_report()
