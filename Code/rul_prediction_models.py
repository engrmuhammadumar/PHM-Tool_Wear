import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CROSS-TEST VALIDATION: Novel Indicators vs. MWUT Baseline")
print("=" * 80)

# YOU MUST PROCESS TESTS 1, 2, 3 FIRST
# This code assumes you've run novel_rul_indicators.py on all three tests

tests = {
    'Test_1': r'E:\RUL\1\novel_indicators_v2.csv',
    'Test_2': r'E:\RUL\2\novel_indicators_v2.csv',
    'Test_3': r'E:\RUL\3\novel_indicators_v2.csv'
}

# Load all tests
data = {}
for test_name, path in tests.items():
    try:
        df = pd.read_csv(path).dropna()
        data[test_name] = df
        print(f"✓ {test_name} loaded: {len(df)} samples")
    except:
        print(f"✗ {test_name} not found - run novel_rul_indicators.py first")
        exit()

# Define feature sets
feature_sets = {
    'Novel_Proposed': ['EWHI', 'CDA'],
    'Traditional_Baseline': [f'ch{i}_rms' for i in range(1, 9)] + 
                           [f'ch{i}_kurtosis' for i in range(1, 9)]
}

# Leave-one-out cross-validation
results = []

for test_out_name in tests.keys():
    print(f"\n{'='*80}")
    print(f"Training on other tests, testing on {test_out_name}")
    print(f"{'='*80}")
    
    # Combine training tests
    train_tests = [name for name in tests.keys() if name != test_out_name]
    df_train = pd.concat([data[name] for name in train_tests])
    df_test = data[test_out_name]
    
    print(f"Training samples: {len(df_train)}")
    print(f"Testing samples: {len(df_test)}")
    
    for feat_name, features in feature_sets.items():
        X_train = df_train[features].values
        y_train = df_train['True_RUL'].values
        X_test = df_test[features].values
        y_test = df_test['True_RUL'].values
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_pred = np.clip(y_pred, 0, None)
        
        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Test_Out': test_out_name,
            'Features': feat_name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        })
        
        print(f"  {feat_name:25s} | RMSE: {rmse:7.2f} | R²: {r2:7.4f}")

# Calculate improvement
results_df = pd.DataFrame(results)
novel_avg = results_df[results_df['Features'] == 'Novel_Proposed']['RMSE'].mean()
baseline_avg = results_df[results_df['Features'] == 'Traditional_Baseline']['RMSE'].mean()
improvement = ((baseline_avg - novel_avg) / baseline_avg) * 100

print(f"\n{'='*80}")
print("FINAL RESULTS")
print(f"{'='*80}")
print(f"Novel Approach Average RMSE: {novel_avg:.2f}")
print(f"Baseline Approach Average RMSE: {baseline_avg:.2f}")
print(f"Improvement: {improvement:.1f}%")

if improvement > 15:
    print("\n✓ PUBLISHABLE: >15% improvement is significant")
elif improvement > 10:
    print("\n⚠ MARGINAL: 10-15% improvement might be publishable with good story")
else:
    print("\n✗ NOT SUFFICIENT: <10% improvement won't impress reviewers")