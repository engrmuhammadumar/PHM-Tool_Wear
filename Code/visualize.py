import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading data for visualization...")
file_path = r'F:\concrete data\test 3\per_file_features_800.csv'
df = pd.read_csv(file_path)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"✓ Data loaded: {len(df)} records, {len(df.columns)} features")

# ============================================================================
# VISUALIZATION 1: DISTRIBUTION OF ALL NUMERIC FEATURES
# ============================================================================
print("\n1. Creating distribution plots for all numeric features...")

if len(numeric_cols) > 0:
    n_cols = 4
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten() if len(numeric_cols) > 1 else [axes]
    
    for idx, col in enumerate(numeric_cols):
        axes[idx].hist(df[col].dropna(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)
        
        # Add statistics text
        mean_val = df[col].mean()
        median_val = df[col].median()
        axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        axes[idx].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        axes[idx].legend()
    
    # Hide extra subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(r'F:\concrete data\test 3\01_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: 01_distributions.png")

# ============================================================================
# VISUALIZATION 2: BOX PLOTS FOR OUTLIER DETECTION
# ============================================================================
print("\n2. Creating box plots for outlier detection...")

if len(numeric_cols) > 0:
    n_cols = 4
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten() if len(numeric_cols) > 1 else [axes]
    
    for idx, col in enumerate(numeric_cols):
        axes[idx].boxplot(df[col].dropna(), vert=True)
        axes[idx].set_title(f'Box Plot: {col}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(col)
        axes[idx].grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(r'F:\concrete data\test 3\02_boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: 02_boxplots.png")

# ============================================================================
# VISUALIZATION 3: CORRELATION HEATMAP
# ============================================================================
print("\n3. Creating correlation heatmap...")

if len(numeric_cols) > 1:
    plt.figure(figsize=(16, 14))
    correlation_matrix = df[numeric_cols].corr()
    
    sns.heatmap(correlation_matrix, 
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm', 
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8})
    
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(r'F:\concrete data\test 3\03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: 03_correlation_heatmap.png")

# ============================================================================
# VISUALIZATION 4: PAIRPLOT (for first 6 numeric features)
# ============================================================================
print("\n4. Creating pairplot for feature relationships...")

if len(numeric_cols) >= 2:
    # Select up to 6 features for pairplot (to avoid overcrowding)
    features_for_pairplot = numeric_cols[:min(6, len(numeric_cols))]
    
    pairplot_data = df[features_for_pairplot].copy()
    
    sns.pairplot(pairplot_data, 
                 diag_kind='kde',
                 plot_kws={'alpha': 0.6, 's': 30},
                 diag_kws={'shade': True})
    
    plt.suptitle('Pairwise Feature Relationships', y=1.01, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(r'F:\concrete data\test 3\04_pairplot.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: 04_pairplot.png")

# ============================================================================
# VISUALIZATION 5: STATISTICAL SUMMARY BAR CHART
# ============================================================================
print("\n5. Creating statistical summary visualization...")

if len(numeric_cols) > 0:
    stats_df = df[numeric_cols].describe().T
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Mean values
    stats_df['mean'].sort_values().plot(kind='barh', ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title('Mean Values by Feature', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Mean Value')
    
    # Standard deviation
    stats_df['std'].sort_values().plot(kind='barh', ax=axes[0, 1], color='coral')
    axes[0, 1].set_title('Standard Deviation by Feature', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Standard Deviation')
    
    # Min values
    stats_df['min'].sort_values().plot(kind='barh', ax=axes[1, 0], color='green')
    axes[1, 0].set_title('Minimum Values by Feature', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Minimum Value')
    
    # Max values
    stats_df['max'].sort_values().plot(kind='barh', ax=axes[1, 1], color='red')
    axes[1, 1].set_title('Maximum Values by Feature', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Maximum Value')
    
    plt.tight_layout()
    plt.savefig(r'F:\concrete data\test 3\05_statistical_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: 05_statistical_summary.png")

# ============================================================================
# VISUALIZATION 6: MISSING DATA VISUALIZATION
# ============================================================================
print("\n6. Creating missing data visualization...")

missing_counts = df.isnull().sum()
if missing_counts.sum() > 0:
    plt.figure(figsize=(12, 6))
    missing_data = missing_counts[missing_counts > 0].sort_values(ascending=False)
    
    plt.bar(range(len(missing_data)), missing_data.values, color='crimson', alpha=0.7)
    plt.xticks(range(len(missing_data)), missing_data.index, rotation=45, ha='right')
    plt.ylabel('Number of Missing Values')
    plt.title('Missing Data by Feature', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(r'F:\concrete data\test 3\06_missing_data.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: 06_missing_data.png")
else:
    print("✓ No missing data to visualize - dataset is complete!")

# ============================================================================
# VISUALIZATION 7: CATEGORICAL FEATURES (if any)
# ============================================================================
if len(categorical_cols) > 0:
    print("\n7. Creating categorical feature visualizations...")
    
    n_cols = 2
    n_rows = int(np.ceil(len(categorical_cols) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6*n_rows))
    axes = axes.flatten() if len(categorical_cols) > 1 else [axes]
    
    for idx, col in enumerate(categorical_cols):
        value_counts = df[col].value_counts()
        axes[idx].bar(range(len(value_counts)), value_counts.values, color='teal', alpha=0.7)
        axes[idx].set_xticks(range(len(value_counts)))
        axes[idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
        axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Count')
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    # Hide extra subplots
    for idx in range(len(categorical_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(r'F:\concrete data\test 3\07_categorical_features.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: 07_categorical_features.png")

print("\n" + "="*80)
print("ALL VISUALIZATIONS COMPLETED!")
print("="*80)
print(f"\nAll plots saved to: F:\\concrete data\\test 3\\")
print("\nGenerated files:")
print("  - 01_distributions.png")
print("  - 02_boxplots.png")
print("  - 03_correlation_heatmap.png")
print("  - 04_pairplot.png")
print("  - 05_statistical_summary.png")
print("  - 06_missing_data.png")
if len(categorical_cols) > 0:
    print("  - 07_categorical_features.png")