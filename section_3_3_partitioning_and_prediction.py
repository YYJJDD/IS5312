"""
IS5312 Mini-Project Section 3.3: Partitioning Data and Predicting Attrition (Match)
This script implements data partitioning and predictive modeling for match prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    roc_auc_score, roc_curve, precision_score, recall_score, 
    f1_score, precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 80)
print("Section 3.3: Partitioning Data and Predicting Attrition (Match)")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("1. DATA LOADING AND PREPROCESSING")
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

# Data Preprocessing
df_processed = df.copy()

# Handle categorical variables
categorical_cols = ['gender', 'samerace', 'match']
label_encoders = {}

for col in categorical_cols:
    if col in df_processed.columns:
        le = LabelEncoder()
        df_processed[col] = df_processed[col].astype(str).str.strip()
        df_processed[col] = df_processed[col].replace('', np.nan)
        if df_processed[col].isnull().sum() > 0:
            mode_val = df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else '0'
            df_processed[col].fillna(mode_val, inplace=True)
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
        print(f"\n✓ Encoded '{col}': {le.classes_}")

# Handle missing values
print("\n\nHandling missing values...")
numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
for col in df_processed.columns:
    if col not in categorical_cols and col not in numerical_cols:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        if df_processed[col].dtype in [np.int64, np.float64]:
            numerical_cols.append(col)

for col in numerical_cols:
    if col in df_processed.columns:
        if df_processed[col].isnull().sum() > 0:
            median_val = df_processed[col].median()
            if pd.notna(median_val):
                df_processed[col].fillna(median_val, inplace=True)
                print(f"  Filled '{col}' with median: {median_val:.2f}")

# Prepare features and target
if 'match' not in df_processed.columns:
    print("Error: Could not find target variable 'match'")
    exit(1)

X = df_processed.drop('match', axis=1)
y = df_processed['match']

# Remove non-numerical columns that weren't encoded
X = X.select_dtypes(include=[np.number])

print(f"\n✓ Features prepared: {X.shape[1]} features")
print(f"✓ Target variable distribution:")
print(y.value_counts().sort_index())
print(f"  Match rate: {y.mean():.2%}")

# ============================================================================
# 2. DATA PARTITIONING
# ============================================================================
print("\n" + "=" * 80)
print("2. DATA PARTITIONING")
print("=" * 80)

# Split the data into training and test sets (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Data partitioned successfully:")
print(f"  Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"  Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
print(f"\n  Training set target distribution:")
print(f"    No Match (0): {sum(y_train == 0)} ({sum(y_train == 0)/len(y_train)*100:.1f}%)")
print(f"    Match (1): {sum(y_train == 1)} ({sum(y_train == 1)/len(y_train)*100:.1f}%)")
print(f"\n  Test set target distribution:")
print(f"    No Match (0): {sum(y_test == 0)} ({sum(y_test == 0)/len(y_test)*100:.1f}%)")
print(f"    Match (1): {sum(y_test == 1)} ({sum(y_test == 1)/len(y_test)*100:.1f}%)")

# Scale features for models that require it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"\n✓ Features scaled using StandardScaler")

# ============================================================================
# 3. MODEL TRAINING AND PREDICTION
# ============================================================================
print("\n" + "=" * 80)
print("3. MODEL TRAINING AND PREDICTION")
print("=" * 80)

# Define multiple models to compare
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=20),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

results = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training {name}...")
    print(f"{'='*60}")
    
    # Use scaled data for models that need it
    if name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    # Store results
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'avg_precision': avg_precision,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Print results
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  Average Precision: {avg_precision:.4f}")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"\n  Confusion Matrix:")
    cm = results[name]['confusion_matrix']
    print(f"    [[{cm[0,0]:5d}  {cm[0,1]:5d}]")
    print(f"     [{cm[1,0]:5d}  {cm[1,1]:5d}]]")
    print(f"    TN: {cm[0,0]}, FP: {cm[0,1]}, FN: {cm[1,0]}, TP: {cm[1,1]}")

# ============================================================================
# 4. MODEL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("4. MODEL COMPARISON")
print("=" * 80)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1'] for m in results.keys()],
    'AUC-ROC': [results[m]['auc'] for m in results.keys()],
    'Avg Precision': [results[m]['avg_precision'] for m in results.keys()],
    'CV Accuracy (Mean)': [results[m]['cv_mean'] for m in results.keys()],
    'CV Accuracy (Std)': [results[m]['cv_std'] for m in results.keys()]
})

# Sort by AUC-ROC
comparison_df = comparison_df.sort_values('AUC-ROC', ascending=False)
print("\n" + comparison_df.to_string(index=False))

# Best model
best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
print(f"\n✓ Best Model: {best_model_name} (based on AUC-ROC)")

# ============================================================================
# 5. DETAILED EVALUATION OF BEST MODEL
# ============================================================================
print("\n" + "=" * 80)
print(f"5. DETAILED EVALUATION: {best_model_name}")
print("=" * 80)

y_pred_best = results[best_model_name]['y_pred']
y_pred_proba_best = results[best_model_name]['y_pred_proba']

print("\nClassification Report:")
# 检查实际存在的类别，避免只有一个类别时的错误
unique_classes = sorted(np.unique(np.concatenate([y_test, y_pred_best])))
if len(unique_classes) == 2:
    target_names = ['No Match', 'Match']
else:
    # 如果只有一个类别，根据实际类别设置名称
    target_names = ['No Match' if 0 in unique_classes else 'Match']
print(classification_report(y_test, y_pred_best, target_names=target_names, labels=unique_classes, zero_division=0))

print("\nConfusion Matrix:")
cm = results[best_model_name]['confusion_matrix']
print(f"                Predicted")
print(f"              No Match  Match")
print(f"Actual No Match   {cm[0,0]:5d}   {cm[0,1]:5d}")
print(f"       Match      {cm[1,0]:5d}   {cm[1,1]:5d}")

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("6. GENERATING VISUALIZATIONS")
print("=" * 80)

# 6.1 ROC Curves
plt.figure(figsize=(12, 8))
for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {result['auc']:.3f})", linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('section_3_3_roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ ROC curves saved as 'section_3_3_roc_curves.png'")

# 6.2 Precision-Recall Curves
plt.figure(figsize=(12, 8))
for name, result in results.items():
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, result['y_pred_proba'])
    plt.plot(recall_curve, precision_curve, label=f"{name} (AP = {result['avg_precision']:.3f})", linewidth=2)

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower left', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('section_3_3_pr_curves.png', dpi=300, bbox_inches='tight')
print("✓ Precision-Recall curves saved as 'section_3_3_pr_curves.png'")

# 6.3 Model Performance Comparison Bar Chart
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
# Map display names to actual keys in results dictionary
metric_key_map = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1-score': 'f1'  # The key in results is 'f1', not 'f1_score'
}
for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]
    # Use the mapping to get the correct key from the display name
    metric_key = metric_key_map.get(metric.lower(), metric.lower().replace('-', '_'))
    values = [results[m][metric_key] for m in comparison_df['Model']]
    bars = ax.barh(range(len(comparison_df)), values, color=plt.cm.viridis(np.linspace(0, 1, len(comparison_df))))
    ax.set_yticks(range(len(comparison_df)))
    ax.set_yticklabels(comparison_df['Model'])
    ax.set_xlabel(metric, fontsize=11)
    ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig('section_3_3_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Metrics comparison chart saved as 'section_3_3_metrics_comparison.png'")

# 6.4 Confusion Matrix Heatmap for Best Model
plt.figure(figsize=(10, 8))
cm_best = results[best_model_name]['confusion_matrix']
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Match', 'Match'],
            yticklabels=['No Match', 'Match'],
            cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig('section_3_3_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrix saved as 'section_3_3_confusion_matrix.png'")

# 6.5 Feature Importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    plt.figure(figsize=(12, 8))
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top 15 Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('section_3_3_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Feature importance plot saved as 'section_3_3_feature_importance.png'")
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))

# ============================================================================
# 7. SUMMARY AND CONCLUSIONS
# ============================================================================
print("\n" + "=" * 80)
print("7. SUMMARY AND CONCLUSIONS")
print("=" * 80)

print(f"\nDataset Information:")
print(f"  Total samples: {len(df)}")
print(f"  Features: {X.shape[1]}")
print(f"  Target variable: Match (Binary Classification)")
print(f"  Match rate: {y.mean():.2%}")

print(f"\nData Partitioning:")
print(f"  Training set: {X_train.shape[0]} samples (80%)")
print(f"  Test set: {X_test.shape[0]} samples (20%)")

print(f"\nBest Model: {best_model_name}")
print(f"  Test Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"  Precision: {results[best_model_name]['precision']:.4f}")
print(f"  Recall: {results[best_model_name]['recall']:.4f}")
print(f"  F1-Score: {results[best_model_name]['f1']:.4f}")
print(f"  AUC-ROC: {results[best_model_name]['auc']:.4f}")
print(f"  Average Precision: {results[best_model_name]['avg_precision']:.4f}")
print(f"  Cross-Validation Accuracy: {results[best_model_name]['cv_mean']:.4f} (+/- {results[best_model_name]['cv_std'] * 2:.4f})")

print(f"\nModel Performance Ranking (by AUC-ROC):")
for idx, row in comparison_df.iterrows():
    print(f"  {idx+1}. {row['Model']}: {row['AUC-ROC']:.4f}")

print("\n" + "=" * 80)
print("Section 3.3 Analysis Complete!")
print("=" * 80)
print("\nGenerated Files:")
print("  - section_3_3_roc_curves.png")
print("  - section_3_3_pr_curves.png")
print("  - section_3_3_metrics_comparison.png")
print("  - section_3_3_confusion_matrix.png")
if hasattr(best_model, 'feature_importances_'):
    print("  - section_3_3_feature_importance.png")

