import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("=" * 80)
print("RUL DATA EXPLORATION - Step 1")
print("=" * 80)

# Load the data
file_path = r'E:\RUL\4\final_features.csv'
print(f"\nLoading data from: {file_path}")

try:
    df = pd.read_csv(file_path)
    print("✓ Data loaded successfully!")
    
    # Basic information
    print("\n" + "=" * 80)
    print("BASIC DATA INFORMATION")
    print("=" * 80)
    print(f"Total number of rows: {len(df)}")
    print(f"Total number of columns: {len(df.columns)}")
    
    # Show column names
    print("\n" + "=" * 80)
    print("COLUMN NAMES")
    print("=" * 80)
    for i, col in enumerate(df.columns, 1):
        print(f"{i:3d}. {col}")
    
    # Show data types
    print("\n" + "=" * 80)
    print("DATA TYPES")
    print("=" * 80)
    print(df.dtypes)
    
    # Show first 10 rows
    print("\n" + "=" * 80)
    print("FIRST 10 ROWS")
    print("=" * 80)
    print(df.head(10))
    
    # Show last 10 rows
    print("\n" + "=" * 80)
    print("LAST 10 ROWS")
    print("=" * 80)
    print(df.tail(10))
    
    # Basic statistics
    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY")
    print("=" * 80)
    print(df.describe())
    
    # Check for missing values
    print("\n" + "=" * 80)
    print("MISSING VALUES CHECK")
    print("=" * 80)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✓ No missing values found!")
    else:
        print("Missing values per column:")
        print(missing[missing > 0])
    
    # Check for any RUL or time-related columns
    print("\n" + "=" * 80)
    print("SEARCHING FOR KEY COLUMNS")
    print("=" * 80)
    
    time_cols = [col for col in df.columns if any(keyword in col.lower() 
                 for keyword in ['time', 'cycle', 'step', 'index'])]
    rul_cols = [col for col in df.columns if 'rul' in col.lower()]
    unit_cols = [col for col in df.columns if any(keyword in col.lower() 
                 for keyword in ['unit', 'id', 'machine', 'engine'])]
    
    print(f"Time-related columns found: {time_cols if time_cols else 'None'}")
    print(f"RUL-related columns found: {rul_cols if rul_cols else 'None'}")
    print(f"Unit/ID columns found: {unit_cols if unit_cols else 'None'}")
    
    # Save a small sample for reference
    print("\n" + "=" * 80)
    print("SAVING SAMPLE DATA")
    print("=" * 80)
    sample_file = r'E:\RUL\4\data_sample.csv'
    df.head(100).to_csv(sample_file, index=False)
    print(f"✓ First 100 rows saved to: {sample_file}")
    
    print("\n" + "=" * 80)
    print("EXPLORATION COMPLETE!")
    print("=" * 80)
    print("\nPlease copy ALL the output above and share it with me.")
    print("This will help me understand your data structure.")
    
except FileNotFoundError:
    print(f"\n❌ ERROR: Could not find the file at {file_path}")
    print("\nPlease check:")
    print("1. The file path is correct")
    print("2. The file exists at that location")
    print("3. You have permission to read the file")
    
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    print("\nPlease share this error message with me.")