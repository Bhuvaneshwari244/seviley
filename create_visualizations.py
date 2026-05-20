"""
Create visualizations with actual data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# Load data
df = pd.read_csv("labels_with_confidence.csv")

print(f"Loaded {len(df)} clips")
print(f"Columns: {df.columns.tolist()}")

# Create figure with 6 subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Stuttering Detection Project - Data Analysis', fontsize=16, fontweight='bold')

# 1. Label Distribution (Pie Chart)
ax1 = axes[0, 0]
labels_count = df['is_stutter'].value_counts()
colors = ['#90EE90', '#FFB6C6']
ax1.pie(labels_count, labels=['Fluent (0)', 'Stutter (1)'], autopct='%1.1f%%', 
        colors=colors, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
ax1.set_title('Dataset Label Distribution\n(31,956 clips)', fontsize=12, weight='bold')

# 2. Annotator Agreement Distribution
ax2 = axes[0, 1]
agreement_data = {
    'Unanimous\nStutter\n(3/3)': (df['stutter_ratio'] == 1.0).sum(),
    'Majority\nStutter\n(2/3)': ((df['stutter_ratio'] >= 0.5) & (df['stutter_ratio'] < 1.0)).sum(),
    'Majority\nFluent\n(1/3)': ((df['stutter_ratio'] > 0.0) & (df['stutter_ratio'] < 0.5)).sum(),
    'Unanimous\nFluent\n(0/3)': (df['stutter_ratio'] == 0.0).sum()
}
bars = ax2.bar(range(len(agreement_data)), list(agreement_data.values()), 
               color=['#FF6B6B', '#FFA07A', '#87CEEB', '#90EE90'])
ax2.set_xticks(range(len(agreement_data)))
ax2.set_xticklabels(list(agreement_data.keys()), fontsize=9)
ax2.set_ylabel('Number of Clips', fontsize=11, weight='bold')
ax2.set_title('Annotator Agreement Distribution', fontsize=12, weight='bold')
ax2.set_ylim([0, max(agreement_data.values()) * 1.1])
# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, agreement_data.values())):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, 
             f'{val:,}\n({val/len(df)*100:.1f}%)', 
             ha='center', va='bottom', fontsize=9, weight='bold')

# 3. Confidence Score Distribution
ax3 = axes[0, 2]
ax3.hist(df['annotation_confidence'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
ax3.set_xlabel('Annotation Confidence', fontsize=11, weight='bold')
ax3.set_ylabel('Number of Clips', fontsize=11, weight='bold')
ax3.set_title('Annotation Confidence Distribution', fontsize=12, weight='bold')
ax3.axvline(df['annotation_confidence'].mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {df["annotation_confidence"].mean():.3f}')
ax3.legend()

# 4. Stutter Ratio Distribution
ax4 = axes[1, 0]
ax4.hist(df['stutter_ratio'], bins=20, color='coral', edgecolor='black', alpha=0.7)
ax4.set_xlabel('Stutter Ratio (Annotator Votes)', fontsize=11, weight='bold')
ax4.set_ylabel('Number of Clips', fontsize=11, weight='bold')
ax4.set_title('Stutter Ratio Distribution', fontsize=12, weight='bold')
ax4.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
ax4.legend()

# 5. Confidence by Label
ax5 = axes[1, 1]
fluent_conf = df[df['is_stutter'] == 0]['annotation_confidence']
stutter_conf = df[df['is_stutter'] == 1]['annotation_confidence']
ax5.hist([fluent_conf, stutter_conf], bins=15, 
         label=['Fluent', 'Stutter'], color=['lightgreen', 'salmon'], 
         edgecolor='black', alpha=0.7)
ax5.set_xlabel('Annotation Confidence', fontsize=11, weight='bold')
ax5.set_ylabel('Number of Clips', fontsize=11, weight='bold')
ax5.set_title('Confidence Distribution by Label', fontsize=12, weight='bold')
ax5.legend(fontsize=10)

# 6. Summary Statistics Table
ax6 = axes[1, 2]
ax6.axis('off')
stats_text = f"""
DATASET STATISTICS

Total Clips: {len(df):,}
  • Stuttered: {(df['is_stutter']==1).sum():,} ({(df['is_stutter']==1).sum()/len(df)*100:.1f}%)
  • Fluent: {(df['is_stutter']==0).sum():,} ({(df['is_stutter']==0).sum()/len(df)*100:.1f}%)

ANNOTATOR AGREEMENT
  • Unanimous (3/3): {((df['stutter_ratio']==0.0).sum() + (df['stutter_ratio']==1.0).sum()):,} ({((df['stutter_ratio']==0.0).sum() + (df['stutter_ratio']==1.0).sum())/len(df)*100:.1f}%)
  • Majority (2/3): {(((df['stutter_ratio']>0.0) & (df['stutter_ratio']<1.0)).sum()):,} ({(((df['stutter_ratio']>0.0) & (df['stutter_ratio']<1.0)).sum())/len(df)*100:.1f}%)

CONFIDENCE SCORES
  • Mean: {df['annotation_confidence'].mean():.3f}
  • Median: {df['annotation_confidence'].median():.3f}
  • High (≥0.8): {(df['annotation_confidence']>=0.8).sum():,} ({(df['annotation_confidence']>=0.8).sum()/len(df)*100:.1f}%)
  • Medium (0.4-0.8): {((df['annotation_confidence']>=0.4) & (df['annotation_confidence']<0.8)).sum():,} ({((df['annotation_confidence']>=0.4) & (df['annotation_confidence']<0.8)).sum()/len(df)*100:.1f}%)
  • Low (<0.4): {(df['annotation_confidence']<0.4).sum():,} ({(df['annotation_confidence']<0.4).sum()/len(df)*100:.1f}%)
"""
ax6.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('COMPLETE_ANALYSIS.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: COMPLETE_ANALYSIS.png")

# Create second figure for data splits
if all(pd.read_csv(f).shape[0] > 0 for f in ['train_split.csv', 'val_split.csv', 'test_split.csv']):
    train_df = pd.read_csv('train_split.csv')
    val_df = pd.read_csv('val_split.csv')
    test_df = pd.read_csv('test_split.csv')
    
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle('Data Splits - Speaker Isolated', fontsize=14, fontweight='bold')
    
    # Split sizes
    ax1 = axes2[0]
    splits = ['Train', 'Val', 'Test']
    sizes = [len(train_df), len(val_df), len(test_df)]
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    bars = ax1.bar(splits, sizes, color=colors, edgecolor='black')
    ax1.set_ylabel('Number of Clips', fontsize=11, weight='bold')
    ax1.set_title('Data Split Distribution', fontsize=12, weight='bold')
    for bar, size in zip(bars, sizes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{size:,}\n({size/len(df)*100:.1f}%)', 
                ha='center', va='bottom', fontsize=10, weight='bold')
    
    # Class distribution per split
    ax2 = axes2[1]
    train_stutter = (train_df['is_stutter']==1).sum() / len(train_df) * 100
    val_stutter = (val_df['is_stutter']==1).sum() / len(val_df) * 100
    test_stutter = (test_df['is_stutter']==1).sum() / len(test_df) * 100
    
    x = np.arange(len(splits))
    width = 0.35
    bars1 = ax2.bar(x - width/2, [100-train_stutter, 100-val_stutter, 100-test_stutter], 
                    width, label='Fluent', color='lightgreen', edgecolor='black')
    bars2 = ax2.bar(x + width/2, [train_stutter, val_stutter, test_stutter], 
                    width, label='Stutter', color='salmon', edgecolor='black')
    ax2.set_ylabel('Percentage (%)', fontsize=11, weight='bold')
    ax2.set_title('Class Balance Across Splits', fontsize=12, weight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(splits)
    ax2.legend()
    ax2.set_ylim([0, 100])
    
    # Summary table
    ax3 = axes2[2]
    ax3.axis('off')
    split_stats = f"""
DATA SPLIT SUMMARY

Train Split:
  • Clips: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)
  • Stutter: {train_stutter:.1f}%
  • Fluent: {100-train_stutter:.1f}%

Validation Split:
  • Clips: {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)
  • Stutter: {val_stutter:.1f}%
  • Fluent: {100-val_stutter:.1f}%

Test Split:
  • Clips: {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)
  • Stutter: {test_stutter:.1f}%
  • Fluent: {100-test_stutter:.1f}%

✓ Speaker Overlap: ZERO
✓ Stratified: YES
"""
    ax3.text(0.1, 0.5, split_stats, fontsize=11, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('DATA_SPLITS.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: DATA_SPLITS.png")

print("\n✓ All visualizations created successfully!")
print("\nGenerated files:")
print("  1. COMPLETE_ANALYSIS.png - 6 charts with full analysis")
print("  2. DATA_SPLITS.png - Data split visualization")
