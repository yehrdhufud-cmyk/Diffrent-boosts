import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

X, y = load_breast_cancer(return_X_y=True)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []




from sklearn.linear_model import LogisticRegression
print("[1] Logistic Regression...", end=" ", flush=True)
start = time.time()
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])
scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
elapsed = time.time() - start
results.append(('Logistic Regression', scores.mean(), scores.std(), elapsed))
print(f"OK {scores.mean():.4f} ({elapsed:.2f}s)")




from sklearn.linear_model import RidgeClassifier
print("[2] Ridge Classifier...", end=" ", flush=True)
start = time.time()
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RidgeClassifier(random_state=42))
])
scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
elapsed = time.time() - start
results.append(('Ridge Classifier', scores.mean(), scores.std(), elapsed))
print(f"OK {scores.mean():.4f} ({elapsed:.2f}s)")




from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
print("[3] LDA...", end=" ", flush=True)
start = time.time()
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearDiscriminantAnalysis())
])
scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
elapsed = time.time() - start
results.append(('LDA', scores.mean(), scores.std(), elapsed))
print(f"OK {scores.mean():.4f} ({elapsed:.2f}s)")




from sklearn.tree import DecisionTreeClassifier
print("[4] Decision Tree...", end=" ", flush=True)
start = time.time()
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', DecisionTreeClassifier(max_depth=10, random_state=42))
])
scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
elapsed = time.time() - start
results.append(('Decision Tree', scores.mean(), scores.std(), elapsed))
print(f"OK {scores.mean():.4f} ({elapsed:.2f}s)")




from sklearn.ensemble import RandomForestClassifier
print("[5] Random Forest...", end=" ", flush=True)
start = time.time()
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])
scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
elapsed = time.time() - start
results.append(('Random Forest', scores.mean(), scores.std(), elapsed))
print(f"OK {scores.mean():.4f} ({elapsed:.2f}s)")


from sklearn.ensemble import ExtraTreesClassifier
print("[6] Extra Trees...", end=" ", flush=True)
start = time.time()
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])
scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
elapsed = time.time() - start
results.append(('Extra Trees', scores.mean(), scores.std(), elapsed))
print(f"OK {scores.mean():.4f} ({elapsed:.2f}s)")




from sklearn.ensemble import BaggingClassifier
print("[7] Bagging...", end=" ", flush=True)
start = time.time()
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', BaggingClassifier(n_estimators=50, random_state=42, n_jobs=-1))
])
scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
elapsed = time.time() - start
results.append(('Bagging', scores.mean(), scores.std(), elapsed))
print(f"OK {scores.mean():.4f} ({elapsed:.2f}s)")




from sklearn.ensemble import AdaBoostClassifier
print("[8] AdaBoost...", end=" ", flush=True)
start = time.time()
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', AdaBoostClassifier(n_estimators=100, random_state=42))
])
scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
elapsed = time.time() - start
results.append(('AdaBoost', scores.mean(), scores.std(), elapsed))
print(f"OK {scores.mean():.4f} ({elapsed:.2f}s)")




from sklearn.ensemble import GradientBoostingClassifier
print("[9] Gradient Boosting...", end=" ", flush=True)
start = time.time()
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingClassifier(n_estimators=100, random_state=42))
])
scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
elapsed = time.time() - start
results.append(('Gradient Boosting', scores.mean(), scores.std(), elapsed))
print(f"OK {scores.mean():.4f} ({elapsed:.2f}s)")




from sklearn.ensemble import HistGradientBoostingClassifier
print("[10] HistGradientBoosting...", end=" ", flush=True)
start = time.time()
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', HistGradientBoostingClassifier(max_iter=100, random_state=42))
])
scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
elapsed = time.time() - start
results.append(('HistGradientBoosting', scores.mean(), scores.std(), elapsed))
print(f"OK {scores.mean():.4f} ({elapsed:.2f}s)")




from sklearn.neighbors import KNeighborsClassifier
print("[11] KNN...", end=" ", flush=True)
start = time.time()
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier(n_neighbors=5))
])
scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
elapsed = time.time() - start
results.append(('KNN', scores.mean(), scores.std(), elapsed))
print(f"OK {scores.mean():.4f} ({elapsed:.2f}s)")




from sklearn.svm import SVC
print("[12] SVM RBF...", end=" ", flush=True)
start = time.time()
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(probability=True, random_state=42))
])
scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
elapsed = time.time() - start
results.append(('SVM RBF', scores.mean(), scores.std(), elapsed))
print(f"OK {scores.mean():.4f} ({elapsed:.2f}s)")


print("[13] SVM Linear...", end=" ", flush=True)
start = time.time()
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(kernel='linear', probability=True, random_state=42))
])
scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
elapsed = time.time() - start
results.append(('SVM Linear', scores.mean(), scores.std(), elapsed))
print(f"OK {scores.mean():.4f} ({elapsed:.2f}s)")




from sklearn.naive_bayes import GaussianNB
print("[14] Gaussian NB...", end=" ", flush=True)
start = time.time()
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GaussianNB())
])
scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
elapsed = time.time() - start
results.append(('Gaussian NB', scores.mean(), scores.std(), elapsed))
print(f"OK {scores.mean():.4f} ({elapsed:.2f}s)")




from sklearn.neural_network import MLPClassifier
print("[15] MLP...", end=" ", flush=True)
start = time.time()
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500,
                            early_stopping=True, random_state=42))
])
scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
elapsed = time.time() - start
results.append(('MLP', scores.mean(), scores.std(), elapsed))
print(f"OK {scores.mean():.4f} ({elapsed:.2f}s)")




try:
    import xgboost as xgb
    print("[16] XGBoost...", end=" ", flush=True)
    start = time.time()
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', xgb.XGBClassifier(n_estimators=100, tree_method='hist',
                                    random_state=42, n_jobs=-1, verbosity=0))
    ])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    elapsed = time.time() - start
    results.append(('XGBoost', scores.mean(), scores.std(), elapsed))
    print(f"OK {scores.mean():.4f} ({elapsed:.2f}s)")
except ImportError:
    print("[16] XGBoost... SKIPPED")


try:
    import lightgbm as lgb
    print("[17] LightGBM...", end=" ", flush=True)
    start = time.time()
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', lgb.LGBMClassifier(n_estimators=100, num_leaves=31,
                                     verbose=-1, random_state=42, n_jobs=-1))
    ])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    elapsed = time.time() - start
    results.append(('LightGBM', scores.mean(), scores.std(), elapsed))
    print(f"OK {scores.mean():.4f} ({elapsed:.2f}s)")
except ImportError:
    print("[17] LightGBM... SKIPPED")


try:
    from catboost import CatBoostClassifier
    print("[18] CatBoost...", end=" ", flush=True)
    start = time.time()
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', CatBoostClassifier(iterations=100, verbose=0,
                                     random_seed=42, thread_count=-1))
    ])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    elapsed = time.time() - start
    results.append(('CatBoost', scores.mean(), scores.std(), elapsed))
    print(f"OK {scores.mean():.4f} ({elapsed:.2f}s)")
except ImportError:
    print("[18] CatBoost... SKIPPED")



df = pd.DataFrame(results, columns=['Model', 'Mean', 'Std', 'Time'])
df = df.sort_values('Mean', ascending=False).reset_index(drop=True)

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(df.to_string(index=False))

print(f"\nBest: {df.iloc[0]['Model']} -> {df.iloc[0]['Mean']:.4f}")


fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].barh(range(len(df)), df['Mean'], xerr=df['Std'],
             color='steelblue', alpha=0.8, capsize=3)
axes[0].set_yticks(range(len(df)))
axes[0].set_yticklabels(df['Model'])
axes[0].set_xlabel('Score')
axes[0].set_title('Accuracy')
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

df_time = df.sort_values('Time')
axes[1].barh(range(len(df_time)), df_time['Time'],
             color='coral', alpha=0.8)
axes[1].set_yticks(range(len(df_time)))
axes[1].set_yticklabels(df_time['Model'])
axes[1].set_xlabel('Time (s)')
axes[1].set_title('Training Time')
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

df.to_csv('results.csv', index=False)
print("\nSaved: results.csv")