"""
IS5312 Mini-Project: Speed Dating Match Prediction
Predicting the likelihood to match based on different factors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the data
print("=" * 80)
print("IS5312 Mini-Project: Speed Dating Match Prediction")
print("=" * 80)

# Define column names based on variables.txt
column_names = [
    'gender', 'age', 'age_o', 'samerace', 'attractive_o', 'sincere_o', 
    'intelligence_o', 'funny_o', 'ambition_o', 'shared_interests_o',
    'attractive_important', 'sincere_important', 'intelligence_important',
    'funny_important', 'ambition_important', 'shared_interests_important',
    'attractive', 'sincere', 'intelligence', 'funny', 'ambition',
    'attractive_partner', 'sincere_partner', 'intelligence_partner',
    'funny_partner', 'ambition_partner', 'shared_interests_partner',
    'interests_correlate', 'expected_num_interested_in_me', 
    'expected_num_matches', 'like', 'guess_prob_liked', 'match'
]

try:
    # Read CSV without header
    df = pd.read_csv('SpeedDating.csv', header=None, names=column_names)
    
    # Clean byte strings from the data
    for col in df.columns:
        if df[col].dtype == 'object':
            # Remove byte string markers
            df[col] = df[col].astype(str).str.replace("b'", "").str.replace("'", "").str.strip()
    
    print(f"\n✓ Data loaded successfully!")
    print(f"  Dataset shape: {df.shape}")
    print(f"  Number of features: {df.shape[1] - 1}")
    print(f"  Number of samples: {df.shape[0]}")
except FileNotFoundError:
    print("Error: SpeedDating.csv not found!")
    exit(1)
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Display basic information
print("\n" + "=" * 80)
print("1. DATA EXPLORATION")
print("=" * 80)

print("\nFirst few rows:")
print(df.head())

print("\n\nColumn names:")
print(df.columns.tolist())

print("\n\nData types:")
print(df.dtypes)

print("\n\nMissing values:")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing Percentage': missing_pct
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
if len(missing_df) > 0:
    print(missing_df)
else:
    print("No missing values found!")

print("\n\nBasic statistics:")
print(df.describe())

# Check target variable
print("\n\nTarget variable distribution (match):")
if 'match' in df.columns:
    print(df['match'].value_counts())
    # Convert to numeric for mean calculation (handle string values)
    match_numeric = pd.to_numeric(df['match'].astype(str).str.replace("b'", "").str.replace("'", "").str.strip(), errors='coerce')
    if match_numeric.notna().any():
        print(f"\nMatch rate: {match_numeric.mean():.2%}")
    else:
        print("\nMatch rate: Could not calculate (non-numeric values)")
else:
    print("Warning: 'match' column not found. Checking for similar column names...")
    print([col for col in df.columns if 'match' in col.lower()])

# Data Preprocessing
print("\n" + "=" * 80)
print("2. DATA PREPROCESSING")
print("=" * 80)

# Create a copy for preprocessing
df_processed = df.copy()

# Handle categorical variables
categorical_cols = ['gender', 'samerace', 'match']
label_encoders = {}

for col in categorical_cols:
    if col in df_processed.columns:
        le = LabelEncoder()
        # Convert to string and clean
        df_processed[col] = df_processed[col].astype(str).str.strip()
        # Handle empty strings and convert to numeric where possible
        df_processed[col] = df_processed[col].replace('', np.nan)
        # Fill NaN with mode for categorical
        if df_processed[col].isnull().sum() > 0:
            mode_val = df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else '0'
            df_processed[col].fillna(mode_val, inplace=True)
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
        print(f"\n✓ Encoded '{col}': {le.classes_}")

# Handle missing values
print("\n\nHandling missing values...")
# Convert numerical columns, handling empty strings
numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
# Also try to convert object columns that should be numeric
for col in df_processed.columns:
    if col not in categorical_cols and col not in numerical_cols:
        # Try to convert to numeric
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        if df_processed[col].dtype in [np.int64, np.float64]:
            numerical_cols.append(col)

# Fill numerical columns with median
for col in numerical_cols:
    if col in df_processed.columns:
        if df_processed[col].isnull().sum() > 0:
            median_val = df_processed[col].median()
            if pd.notna(median_val):
                df_processed[col].fillna(median_val, inplace=True)
                print(f"  Filled '{col}' with median: {median_val:.2f}")

# Prepare features and target
if 'match' in df_processed.columns:
    X = df_processed.drop('match', axis=1)
    y = df_processed['match']
    
    # Remove non-numerical columns that weren't encoded
    X = X.select_dtypes(include=[np.number])
    
    print(f"\n✓ Features prepared: {X.shape[1]} features")
    print(f"✓ Target variable: {y.value_counts().to_dict()}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n✓ Data split:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"✓ Features scaled using StandardScaler")
    
else:
    print("Error: Could not find target variable 'match'")
    exit(1)

# Model Building
print("\n" + "=" * 80)
print("3. MODEL BUILDING")
print("=" * 80)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Use scaled data for Logistic Regression, original for tree-based models
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        # Cross-validation on scaled data
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        # Cross-validation on original data
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'auc': auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Model Comparison
print("\n" + "=" * 80)
print("4. MODEL COMPARISON")
print("=" * 80)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Test Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'AUC-ROC': [results[m]['auc'] for m in results.keys()],
    'CV Accuracy (Mean)': [results[m]['cv_mean'] for m in results.keys()],
    'CV Accuracy (Std)': [results[m]['cv_std'] for m in results.keys()]
})

comparison_df = comparison_df.sort_values('AUC-ROC', ascending=False)
print("\n" + comparison_df.to_string(index=False))

# Best model
best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
print(f"\n✓ Best Model: {best_model_name}")

# Detailed evaluation of best model
print("\n" + "=" * 80)
print(f"5. DETAILED EVALUATION: {best_model_name}")
print("=" * 80)

y_pred_best = results[best_model_name]['y_pred']
y_pred_proba_best = results[best_model_name]['y_pred_proba']

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['No Match', 'Match']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)
print(f"\nTrue Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

# Feature Importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    print("\n" + "=" * 80)
    print("6. FEATURE IMPORTANCE")
    print("=" * 80)
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15).to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance')
    plt.title(f'Top 15 Feature Importance - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Feature importance plot saved as 'feature_importance.png'")

# ROC Curve
print("\n" + "=" * 80)
print("7. ROC CURVE")
print("=" * 80)

plt.figure(figsize=(10, 8))
for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {result['auc']:.3f})", linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Model Comparison')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ ROC curves plot saved as 'roc_curves.png'")

# Summary Statistics
print("\n" + "=" * 80)
print("8. SUMMARY")
print("=" * 80)

print(f"\nDataset: Speed Dating Match Prediction")
print(f"Total samples: {len(df)}")
print(f"Features: {X.shape[1]}")
print(f"Target variable: Match (Binary Classification)")
print(f"\nBest Model: {best_model_name}")
print(f"  Test Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"  AUC-ROC: {results[best_model_name]['auc']:.4f}")
print(f"  Cross-Validation Accuracy: {results[best_model_name]['cv_mean']:.4f} (+/- {results[best_model_name]['cv_std'] * 2:.4f})")

print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)
