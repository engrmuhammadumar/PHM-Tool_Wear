import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# ============================================================================
# STEP 1: LOAD THE DATA
# ============================================================================
print("="*80)
print("LOADING CONCRETE RUN-TO-FAILURE DATA")
print("="*80)

file_path = r'F:\concrete data\test 3\per_file_features_800.csv'
df = pd.read_csv(file_path)

print(f"\n✓ Data loaded successfully!")
print(f"✓ Total records: {len(df)}")
print(f"✓ Total features: {len(df.columns)}")

# ============================================================================
# STEP 2: BASIC DATA STRUCTURE
# ============================================================================
print("\n" + "="*80)
print("STEP 2: DATA STRUCTURE OVERVIEW")
print("="*80)

print("\n--- Column Names and Data Types ---")
print(df.dtypes)

print("\n--- First 5 Records ---")
print(df.head())

print("\n--- Last 5 Records ---")
print(df.tail())

print("\n--- Dataset Shape ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# ============================================================================
# STEP 3: DATA QUALITY ASSESSMENT
# ============================================================================
print("\n" + "="*80)
print("STEP 3: DATA QUALITY ASSESSMENT")
print("="*80)

print("\n--- Missing Values Analysis ---")
missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
if len(missing_data) > 0:
    print(missing_data.to_string(index=False))
else:
    print("✓ No missing values found!")

print("\n--- Duplicate Records ---")
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

print("\n--- Data Types Summary ---")
print(df.dtypes.value_counts())

# ============================================================================
# STEP 4: STATISTICAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("STEP 4: COMPREHENSIVE STATISTICAL SUMMARY")
print("="*80)

# Separate numeric and non-numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"\n--- Numeric Features ({len(numeric_cols)}) ---")
if len(numeric_cols) > 0:
    print(df[numeric_cols].describe().T)
    
    print("\n--- Additional Statistics for Numeric Features ---")
    additional_stats = pd.DataFrame({
        'Skewness': df[numeric_cols].skew(),
        'Kurtosis': df[numeric_cols].kurtosis(),
        'Range': df[numeric_cols].max() - df[numeric_cols].min(),
        'IQR': df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25)
    })
    print(additional_stats)

print(f"\n--- Categorical Features ({len(categorical_cols)}) ---")
if len(categorical_cols) > 0:
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())
        print(f"Unique values: {df[col].nunique()}")

# ============================================================================
# STEP 5: OUTLIER DETECTION
# ============================================================================
print("\n" + "="*80)
print("STEP 5: OUTLIER DETECTION (IQR Method)")
print("="*80)

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound

if len(numeric_cols) > 0:
    outlier_summary = []
    for col in numeric_cols:
        n_outliers, lower, upper = detect_outliers_iqr(df, col)
        outlier_summary.append({
            'Feature': col,
            'Outliers_Count': n_outliers,
            'Outliers_Percentage': round(n_outliers/len(df)*100, 2),
            'Lower_Bound': round(lower, 4),
            'Upper_Bound': round(upper, 4)
        })
    outlier_df = pd.DataFrame(outlier_summary)
    outlier_df = outlier_df[outlier_df['Outliers_Count'] > 0].sort_values('Outliers_Count', ascending=False)
    if len(outlier_df) > 0:
        print(outlier_df.to_string(index=False))
    else:
        print("✓ No outliers detected in any numeric feature!")

# ============================================================================
# STEP 6: CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STEP 6: CORRELATION ANALYSIS")
print("="*80)

if len(numeric_cols) > 1:
    print("\n--- Correlation Matrix (Pearson) ---")
    correlation_matrix = df[numeric_cols].corr()
    print(correlation_matrix)
    
    print("\n--- Top 10 Strongest Correlations ---")
    # Get upper triangle of correlation matrix
    corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_pairs.append({
                'Feature_1': correlation_matrix.columns[i],
                'Feature_2': correlation_matrix.columns[j],
                'Correlation': correlation_matrix.iloc[i, j]
            })
    corr_pairs_df = pd.DataFrame(corr_pairs)
    corr_pairs_df['Abs_Correlation'] = corr_pairs_df['Correlation'].abs()
    top_correlations = corr_pairs_df.sort_values('Abs_Correlation', ascending=False).head(10)
    print(top_correlations[['Feature_1', 'Feature_2', 'Correlation']].to_string(index=False))

# ============================================================================
# STEP 7: SAVE ANALYSIS REPORT
# ============================================================================
print("\n" + "="*80)
print("STEP 7: SAVING ANALYSIS REPORT")
print("="*80)

report_path = r'F:\concrete data\test 3\data_analysis_report.txt'
with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("CONCRETE RUN-TO-FAILURE DATA ANALYSIS REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(f"Total Records: {len(df)}\n")
    f.write(f"Total Features: {len(df.columns)}\n")
    f.write(f"Numeric Features: {len(numeric_cols)}\n")
    f.write(f"Categorical Features: {len(categorical_cols)}\n\n")
    f.write("Column Names:\n")
    for col in df.columns:
        f.write(f"  - {col}\n")

print(f"✓ Analysis report saved to: {report_path}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nNext steps: Run the visualization script to create charts and graphs!")