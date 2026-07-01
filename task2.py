"""
============================================================
  TASK 2 — CREDIT CARD FRAUD DETECTION
  Dataset: Kaggle - kartik2112/fraud-detection
  Models: Logistic Regression, Decision Tree, Random Forest
============================================================
"""

# ── 1. IMPORTS ──────────────────────────────────────────────
import pandas as pd
import numpy as np
import re
import warnings
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score
)

warnings.filterwarnings('ignore')

print("=" * 60)
print("  CREDIT CARD FRAUD DETECTION — ML Pipeline")
print("=" * 60)


# ── 2. LOAD DATASET ─────────────────────────────────────────
train_df = pd.read_csv(r'C:\Users\panda\Downloads\archive (1)\fraudTrain.csv')
test_df  = pd.read_csv(r'C:\Users\panda\Downloads\archive (1)\fraudTest.csv')

df = pd.concat([train_df, test_df], ignore_index=True)

print(f"\n✅  Dataset loaded: {len(df):,} transactions")
print(f"    Fraud cases   : {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
print(f"    Legit cases   : {(df['is_fraud']==0).sum():,}")
print(f"    Columns       : {list(df.columns)}")


# ── 3. FEATURE ENGINEERING ──────────────────────────────────
# Extract time features from trans_date_trans_time
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['hour']      = df['trans_date_trans_time'].dt.hour
df['day']       = df['trans_date_trans_time'].dt.day
df['month']     = df['trans_date_trans_time'].dt.month
df['dayofweek'] = df['trans_date_trans_time'].dt.dayofweek

# Age from dob
df['dob'] = pd.to_datetime(df['dob'])
df['age'] = (pd.Timestamp('2024-01-01') - df['dob']).dt.days // 365

# Distance between merchant and customer
df['distance'] = np.sqrt(
    (df['lat'] - df['merch_lat'])**2 +
    (df['long'] - df['merch_long'])**2
)

# Encode categorical columns
le = LabelEncoder()
for col in ['category', 'gender']:
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))

print("✅  Feature engineering done.")


# ── 4. SELECT FEATURES ──────────────────────────────────────
features = [
    'amt', 'hour', 'day', 'month', 'dayofweek',
    'age', 'distance', 'city_pop',
    'category_enc', 'gender_enc'
]

X = df[features]
y = df['is_fraud']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅  Split: {len(X_train):,} train | {len(X_test):,} test")


# ── 5. TRAIN MODELS ─────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=8,
                                                   random_state=42, n_jobs=-1)
}

results = {}
print("\n── Training Models ──────────────────────────────────────")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    results[name] = {
        'model': model, 'pred': y_pred, 'prob': y_prob,
        'acc': acc, 'f1': f1, 'prec': prec, 'rec': rec, 'auc': auc
    }
    print(f"  {name:<22}  Acc={acc:.2%}  F1={f1:.2%}  AUC={auc:.2%}")


# ── 6. BEST MODEL ───────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['auc'])
best      = results[best_name]
print(f"\n🏆  Best Model: {best_name}")
print("\nClassification Report:")
print(classification_report(y_test, best['pred'],
                            target_names=['Legitimate', 'Fraud']))


# ── 7. VISUALISATION ────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#0D1117')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
tkw = dict(color='white', fontsize=12, fontweight='bold', pad=10)

# A: Fraud vs Legit Distribution
ax0 = fig.add_subplot(gs[0, 0])
ax0.set_facecolor('#161B22')
counts = [( y == 0).sum(), (y == 1).sum()]
labels = ['Legitimate', 'Fraud']
colors = ['#2ECC71', '#E74C3C']
bars   = ax0.bar(labels, counts, color=colors, edgecolor='none', width=0.5)
for bar, val in zip(bars, counts):
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
             f"{val:,}", ha='center', color='white', fontsize=11)
ax0.set_title("Transaction Distribution", **tkw)
ax0.set_ylabel("Count", color='#8B949E')
ax0.tick_params(colors='#8B949E')
ax0.spines[:].set_visible(False)

# B: Model Comparison (F1, AUC)
ax1 = fig.add_subplot(gs[0, 1])
ax1.set_facecolor('#161B22')
mnames = list(results.keys())
f1s  = [results[m]['f1']  for m in mnames]
aucs = [results[m]['auc'] for m in mnames]
x, w = np.arange(len(mnames)), 0.35
b1 = ax1.bar(x - w/2, f1s,  w, label='F1 Score', color='#3498DB', edgecolor='none')
b2 = ax1.bar(x + w/2, aucs, w, label='AUC',      color='#E74C3C', edgecolor='none')
for bars in [b1, b2]:
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{bar.get_height():.2f}", ha='center', color='white', fontsize=8)
ax1.set_title("Model Comparison", **tkw)
ax1.set_xticks(x)
ax1.set_xticklabels(['Logistic\nRegr.', 'Decision\nTree', 'Random\nForest'],
                    color='#8B949E', fontsize=9)
ax1.set_ylim(0, 1.12)
ax1.tick_params(colors='#8B949E')
ax1.spines[:].set_visible(False)
ax1.legend(facecolor='#161B22', labelcolor='white', fontsize=9)

# C: Confusion Matrix
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor('#161B22')
cm = confusion_matrix(y_test, best['pred'])
im = ax2.imshow(cm, cmap='Reds')
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Legitimate', 'Fraud'], color='#8B949E')
ax2.set_yticklabels(['Legitimate', 'Fraud'], color='#8B949E')
for i in range(2):
    for j in range(2):
        ax2.text(j, i, f"{cm[i,j]:,}", ha='center', va='center',
                 color='white', fontsize=13, fontweight='bold')
ax2.set_title(f"Confusion Matrix\n({best_name})", **tkw)
ax2.set_xlabel("Predicted", color='#8B949E')
ax2.set_ylabel("Actual",    color='#8B949E')

# D: Fraud by Hour
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor('#161B22')
fraud_hour = df[df['is_fraud'] == 1]['hour'].value_counts().sort_index()
ax3.bar(fraud_hour.index, fraud_hour.values, color='#E74C3C', edgecolor='none')
ax3.set_title("Fraud by Hour of Day", **tkw)
ax3.set_xlabel("Hour", color='#8B949E')
ax3.set_ylabel("Fraud Count", color='#8B949E')
ax3.tick_params(colors='#8B949E')
ax3.spines[:].set_visible(False)

# E: Fraud by Category
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor('#161B22')
fraud_cat = df[df['is_fraud'] == 1]['category'].value_counts().head(8)
ax4.barh(fraud_cat.index, fraud_cat.values, color='#E67E22', edgecolor='none')
ax4.set_title("Top Fraud Categories", **tkw)
ax4.set_xlabel("Count", color='#8B949E')
ax4.tick_params(colors='#8B949E')
ax4.spines[:].set_visible(False)

# F: Feature Importance (Random Forest)
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor('#161B22')
rf_model = results['Random Forest']['model']
importances = rf_model.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=True)
colors_fi = ['#E74C3C' if v > feat_imp.median() else '#3498DB' for v in feat_imp]
ax5.barh(feat_imp.index, feat_imp.values, color=colors_fi, edgecolor='none')
ax5.set_title("Feature Importance\n(Random Forest)", **tkw)
ax5.set_xlabel("Importance", color='#8B949E')
ax5.tick_params(colors='#8B949E')
ax5.spines[:].set_visible(False)

fig.suptitle("💳  Credit Card Fraud Detection — ML Pipeline",
             color='white', fontsize=16, fontweight='bold', y=0.98)

import os
desktop   = os.path.join(os.path.expanduser("~"), "Desktop")
save_path = os.path.join(desktop, "fraud_detection_result.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')

plt.show()

print("\n" + "="*60)
print(f"  ✅  Best Model : {best_name}")
print(f"  ✅  Accuracy   : {best['acc']:.2%}")
print(f"  ✅  F1 Score   : {best['f1']:.2%}")
print(f"  ✅  AUC Score  : {best['auc']:.2%}")
print(f"  ✅  Precision  : {best['prec']:.2%}")
print(f"  ✅  Recall     : {best['rec']:.2%}")
print("="*60)
