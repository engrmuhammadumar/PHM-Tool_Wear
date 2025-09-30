import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD DATA
# ============================================================================
print("="*80)
print("ADVANCED CONCRETE DATA ANALYSIS")
print("="*80)

file_path = r'F:\concrete data\test 3\per_file_features_800.csv'
df = pd.read_csv(file_path)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n✓ Data loaded: {len(df)} records, {len(numeric_cols)} numeric features")

# ============================================================================
# ANALYSIS 1: FEATURE IMPORTANCE BASED ON VARIANCE
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 1: FEATURE VARIANCE ANALYSIS")
print("="*80)

variance_df = pd.DataFrame({
    'Feature': numeric_cols,
    'Variance': [df[col].var() for col in numeric_cols],
    'Std_Dev': [df[col].std() for col in numeric_cols],
    'Coefficient_of_Variation': [df[col].std() / df[col].mean() if df[col].mean() != 0 else 0 for col in numeric_cols]
})
variance_df = variance_df.sort_values('Variance', ascending=False)
print("\n--- Top 10 Features by Variance ---")
print(variance_df.head(10).to_string(index=False))

# Visualize
plt.figure(figsize=(14, 6))
top_features = variance_df.head(15)
plt.bar(range(len(top_features)), top_features['Variance'].values, color='steelblue', alpha=0.7)
plt.xticks(range(len(top_features)), top_features['Feature'].values, rotation=45, ha='right')
plt.ylabel('Variance')
plt.title('Top 15 Features by Variance', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(r'F:\concrete data\test 3\08_feature_variance.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: 08_feature_variance.png")

# ============================================================================
# ANALYSIS 2: PRINCIPAL COMPONENT ANALYSIS (PCA)
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 2: PRINCIPAL COMPONENT ANALYSIS")
print("="*80)

# Prepare data for PCA
df_numeric = df[numeric_cols].dropna()

if len(df_numeric) > 0 and len(numeric_cols) > 1:
    # Standardize the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    
    # Apply PCA
    n_components = min(10, len(numeric_cols))
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    
    # Explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print(f"\n--- PCA Results ---")
    print(f"Number of components: {n_components}")
    print(f"\nExplained variance by component:")
    for i, (ev, cv) in enumerate(zip(explained_variance, cumulative_variance)):
        print(f"  PC{i+1}: {ev*100:.2f}% (Cumulative: {cv*100:.2f}%)")
    
    # Visualization: Scree plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Individual variance
    axes[0].bar(range(1, n_components+1), explained_variance*100, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance (%)')
    axes[0].set_title('Scree Plot - Individual Variance', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Cumulative variance
    axes[1].plot(range(1, n_components+1), cumulative_variance*100, marker='o', 
                 linewidth=2, markersize=8, color='coral')
    axes[1].axhline(y=80, color='red', linestyle='--', label='80% threshold')
    axes[1].axhline(y=90, color='green', linestyle='--', label='90% threshold')
    axes[1].set_xlabel('Principal Component')
    axes[1].set_ylabel('Cumulative Explained Variance (%)')
    axes[1].set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(r'F:\concrete data\test 3\09_pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: 09_pca_analysis.png")
    
    # PCA Biplot for first two components
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(12, 10))
        
        # Plot samples
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5, s=30, c='steelblue')
        
        # Plot feature vectors
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        for i, feature in enumerate(numeric_cols[:20]):  # Show first 20 features to avoid clutter
            plt.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3,
                     head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.6)
            plt.text(loadings[i, 0]*3.2, loadings[i, 1]*3.2, feature, 
                    fontsize=8, ha='center', va='center')
        
        plt.xlabel(f'PC1 ({explained_variance[0]*100:.2f}%)', fontsize=12)
        plt.ylabel(f'PC2 ({explained_variance[1]*100:.2f}%)', fontsize=12)
        plt.title(f'K-Means Clustering (k={optimal_k}) on PCA Space', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(r'F:\concrete data\test 3\12_kmeans_clusters.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Saved: 12_kmeans_clusters.png")
    
    # Cluster statistics
    df_with_clusters = df_numeric.copy()
    df_with_clusters['Cluster'] = clusters
    
    print("\n--- Cluster Statistics (Mean Values) ---")
    cluster_means = df_with_clusters.groupby('Cluster').mean()
    print(cluster_means.T.head(10))

# ============================================================================
# ANALYSIS 4: HIERARCHICAL CLUSTERING
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 4: HIERARCHICAL CLUSTERING")
print("="*80)

if len(df_numeric) > 0 and len(df_numeric) <= 1000:  # Only for reasonable sample sizes
    # Sample data if too large
    sample_size = min(100, len(scaled_data))
    sample_indices = np.random.choice(len(scaled_data), sample_size, replace=False)
    sample_data = scaled_data[sample_indices]
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(sample_data, method='ward')
    
    # Plot dendrogram
    plt.figure(figsize=(16, 8))
    dendrogram(linkage_matrix, 
               truncate_mode='lastp',
               p=30,
               leaf_rotation=90,
               leaf_font_size=10,
               show_contracted=True)
    plt.xlabel('Sample Index or (Cluster Size)', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.title(f'Hierarchical Clustering Dendrogram (Sample of {sample_size} records)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(r'F:\concrete data\test 3\13_dendrogram.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: 13_dendrogram.png")
else:
    print("Dataset too large for dendrogram visualization (skipping)")

# ============================================================================
# ANALYSIS 5: NORMALITY TESTS
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 5: NORMALITY TESTS (Shapiro-Wilk)")
print("="*80)

normality_results = []
for col in numeric_cols[:20]:  # Test first 20 features
    data_sample = df[col].dropna()
    if len(data_sample) > 3:
        # Use smaller sample for large datasets
        if len(data_sample) > 5000:
            data_sample = data_sample.sample(5000, random_state=42)
        
        statistic, p_value = stats.shapiro(data_sample)
        normality_results.append({
            'Feature': col,
            'Statistic': statistic,
            'P_Value': p_value,
            'Is_Normal': 'Yes' if p_value > 0.05 else 'No'
        })

normality_df = pd.DataFrame(normality_results)
print("\n--- Normality Test Results (α = 0.05) ---")
print(normality_df.to_string(index=False))

# Visualize normality
normal_count = (normality_df['Is_Normal'] == 'Yes').sum()
not_normal_count = (normality_df['Is_Normal'] == 'No').sum()

plt.figure(figsize=(10, 6))
plt.bar(['Normal', 'Not Normal'], [normal_count, not_normal_count], 
        color=['green', 'red'], alpha=0.7)
plt.ylabel('Number of Features')
plt.title('Distribution Normality Summary', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(r'F:\concrete data\test 3\14_normality_summary.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: 14_normality_summary.png")

# ============================================================================
# ANALYSIS 6: FEATURE CORRELATION WITH TARGET (if exists)
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 6: FEATURE RELATIONSHIPS")
print("="*80)

# Check if there's a potential target variable (often named 'target', 'label', 'failure', etc.)
potential_targets = [col for col in df.columns if any(keyword in col.lower() 
                     for keyword in ['target', 'label', 'failure', 'class', 'output', 'result'])]

if potential_targets:
    print(f"\n--- Potential target variable(s) found: {potential_targets} ---")
    
    for target in potential_targets:
        if target in numeric_cols:
            # Calculate correlation with target
            correlations = df[numeric_cols].corr()[target].drop(target).sort_values(key=abs, ascending=False)
            
            print(f"\n--- Top 15 Features Correlated with '{target}' ---")
            print(correlations.head(15))
            
            # Visualize
            plt.figure(figsize=(12, 8))
            top_corr = correlations.head(20)
            colors = ['green' if x > 0 else 'red' for x in top_corr.values]
            plt.barh(range(len(top_corr)), top_corr.values, color=colors, alpha=0.7)
            plt.yticks(range(len(top_corr)), top_corr.index)
            plt.xlabel('Correlation Coefficient')
            plt.title(f'Top 20 Features Correlated with {target}', fontsize=14, fontweight='bold')
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig(f'F:\\concrete data\\test 3\\15_target_correlation_{target}.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            print(f"✓ Saved: 15_target_correlation_{target}.png")
else:
    print("\n⚠ No obvious target variable found.")
    print("Showing strongest pairwise correlations instead:")
    
    # Show strongest correlations
    corr_matrix = df[numeric_cols].corr()
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append({
                'Feature_1': corr_matrix.columns[i],
                'Feature_2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })
    
    corr_df = pd.DataFrame(corr_pairs)
    corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
    top_pairs = corr_df.sort_values('Abs_Correlation', ascending=False).head(15)
    print("\n--- Top 15 Strongest Feature Correlations ---")
    print(top_pairs[['Feature_1', 'Feature_2', 'Correlation']].to_string(index=False))

# ============================================================================
# ANALYSIS 7: DATA QUALITY SCORE
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 7: DATA QUALITY ASSESSMENT")
print("="*80)

quality_metrics = {
    'Total_Records': len(df),
    'Total_Features': len(df.columns),
    'Numeric_Features': len(numeric_cols),
    'Completeness': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
    'Duplicate_Records': df.duplicated().sum(),
    'Duplicate_Percentage': (df.duplicated().sum() / len(df)) * 100
}

print("\n--- Data Quality Metrics ---")
for metric, value in quality_metrics.items():
    if 'Percentage' in metric or 'Completeness' in metric:
        print(f"{metric}: {value:.2f}%")
    else:
        print(f"{metric}: {value}")

# Calculate quality score (0-100)
quality_score = (
    (quality_metrics['Completeness'] * 0.4) +  # 40% weight on completeness
    ((1 - min(quality_metrics['Duplicate_Percentage']/10, 1)) * 30) +  # 30% weight on uniqueness
    (min(len(df)/1000, 1) * 15) +  # 15% weight on sample size
    (min(len(numeric_cols)/50, 1) * 15)  # 15% weight on feature richness
)

print(f"\n{'='*50}")
print(f"OVERALL DATA QUALITY SCORE: {quality_score:.2f}/100")
print(f"{'='*50}")

if quality_score >= 90:
    print("✓ EXCELLENT: Dataset is high quality and ready for analysis")
elif quality_score >= 75:
    print("✓ GOOD: Dataset is suitable for analysis with minor considerations")
elif quality_score >= 60:
    print("⚠ FAIR: Dataset may require some cleaning or preprocessing")
else:
    print("⚠ POOR: Dataset requires significant preprocessing")

# ============================================================================
# SAVE ADVANCED ANALYSIS REPORT
# ============================================================================
print("\n" + "="*80)
print("SAVING ADVANCED ANALYSIS REPORT")
print("="*80)

report_path = r'F:\concrete data\test 3\advanced_analysis_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("CONCRETE RUN-TO-FAILURE DATA - ADVANCED ANALYSIS REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write("DATA QUALITY METRICS\n")
    f.write("-"*40 + "\n")
    for metric, value in quality_metrics.items():
        if 'Percentage' in metric or 'Completeness' in metric:
            f.write(f"{metric}: {value:.2f}%\n")
        else:
            f.write(f"{metric}: {value}\n")
    f.write(f"\nOverall Quality Score: {quality_score:.2f}/100\n\n")
    
    f.write("TOP 10 FEATURES BY VARIANCE\n")
    f.write("-"*40 + "\n")
    f.write(variance_df.head(10).to_string(index=False))
    f.write("\n\n")
    
    if len(normality_results) > 0:
        f.write("NORMALITY TEST SUMMARY\n")
        f.write("-"*40 + "\n")
        f.write(normality_df.to_string(index=False))
        f.write("\n\n")
    
    f.write("="*80 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*80 + "\n")

print(f"✓ Advanced analysis report saved to: {report_path}")

print("\n" + "="*80)
print("ADVANCED ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated visualizations:")
print("  - 08_feature_variance.png")
print("  - 09_pca_analysis.png")
print("  - 10_pca_biplot.png")
print("  - 11_elbow_method.png")
print("  - 12_kmeans_clusters.png")
print("  - 13_dendrogram.png")
print("  - 14_normality_summary.png")
if potential_targets:
    print(f"  - 15_target_correlation_{potential_targets[0]}.png")
print("\nYou are now an expert on your concrete data!")
=12)
        plt.title('PCA Biplot - First Two Components', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(r'F:\concrete data\test 3\10_pca_biplot.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Saved: 10_pca_biplot.png")

# ============================================================================
# ANALYSIS 3: CLUSTERING ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 3: K-MEANS CLUSTERING")
print("="*80)

if len(df_numeric) > 0 and len(numeric_cols) > 1:
    # Elbow method to find optimal number of clusters
    inertias = []
    K_range = range(2, min(11, len(df_numeric)))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        inertias.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, marker='o', linewidth=2, markersize=8, color='steelblue')
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
    plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(r'F:\concrete data\test 3\11_elbow_method.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: 11_elbow_method.png")
    
    # Apply K-means with optimal k (let's use 3 as default)
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    
    print(f"\n--- Clustering Results (k={optimal_k}) ---")
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    print("Samples per cluster:")
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count} samples ({count/len(clusters)*100:.2f}%)")
    
    # Visualize clusters on first two PCs
    if n_components >= 2:
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                            c=clusters, cmap='viridis', s=50, alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(f'PC1 ({explained_variance[0]*100:.2f}%)', fontsize=12)
        plt.ylabel(f'PC2 ({explained_variance[1]*100:.2f}%)', fontsize