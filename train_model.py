import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load data
csv_path = 'startup data.csv'
df = pd.read_csv(csv_path, encoding='utf-8')

# Features expected by the model (25)
features = [
 'age_first_funding_year','age_last_funding_year',
 'age_first_milestone_year','age_last_milestone_year',
 'relationships','funding_rounds','funding_total_usd',
 'milestones','avg_participants',
 'is_CA','is_NY','is_MA','is_TX','is_otherstate',
 'is_software','is_web','is_mobile','is_enterprise',
 'is_advertising','is_gamesvideo','is_ecommerce',
 'is_biotech','is_consulting','is_othercategory',
 'is_top500'
]

# Ensure all features exist in df; if missing, create zero-filled columns
for col in features:
    if col not in df.columns:
        df[col] = 0

# Prepare X and y
X = df[features].fillna(0)
# Map status to binary (acquired -> 1, others -> 0)
if 'status' not in df.columns:
    raise SystemExit('Column "status" not found in CSV')
y = df['status'].astype(str).str.lower().map(lambda s: 1 if s == 'acquired' else 0)

print('Selected features count:', X.shape[1])
print('Sample feature names:', X.columns.tolist()[:10], '...')

# Train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Save model
out = 'random_forest_model.pkl'
joblib.dump(clf, out)
print('Model saved to', out)
