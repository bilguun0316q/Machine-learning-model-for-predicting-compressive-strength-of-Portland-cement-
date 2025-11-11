import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error

# --------------------------------------------------
# 1. Load and prepare data
# --------------------------------------------------

DATA_PATH = r"file/path"
df = pd.read_csv(DATA_PATH)

# Drop missing rows (optional cleanup)
df = df.dropna().reset_index(drop=True)

# Function to categorize cooling rate values into named regimes
def label_cooling(rate):
    if np.isclose(rate, 2.0):
        return "very_slow_2"
    elif np.isclose(rate, 10.0):
        return "slow_10"
    elif np.isclose(rate, 20.0):
        return "moderate_20"
    elif np.isclose(rate, 50.0):
        return "reference_50"
    elif np.isclose(rate, 80.0):
        return "fast_80"
    elif np.isclose(rate, 100.0):
        return "very_fast_100"
    else:
        return f"other_{rate}"

# Apply cooling regime labeling
df["cooling_regime_label"] = df["cooling_rate_C_per_min"].apply(label_cooling)

# Select feature and target columns
feature_cols = [
    "C3S_wt", "C2S_wt", "C3A_wt", "C4AF_wt",
    "alite_grain_um", "belite_grain_um",
    "belite_alite_ratio", "C2S_polymorphic_ratio",
    "cooling_rate_C_per_min", "age_days"
]
target_col = "strength_MPa"

X = df[feature_cols].values
y = df[target_col].values
indices = np.arange(len(df))

# --------------------------------------------------
# 2. Train/Test split
# --------------------------------------------------

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices, test_size=0.2, random_state=42, shuffle=True
)

# --------------------------------------------------
# 3. Random Forest model setup
# --------------------------------------------------

rf = RandomForestRegressor(
    n_estimators=500,      # number of trees
    max_depth=None,        # unlimited depth
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1              # use all CPU cores
)

# --------------------------------------------------
# 4. 10-fold cross-validation
# --------------------------------------------------

cv = KFold(n_splits=10, shuffle=True, random_state=42)

# R2 score
cv_scores_r2 = cross_val_score(rf, X, y, cv=cv, scoring="r2", n_jobs=-1)

# RMSE score
cv_scores_mse = cross_val_score(rf, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
cv_scores_rmse = np.sqrt(-cv_scores_mse)

print("=== 10-fold Cross Validation ===")
print(f"R2  mean ± std : {cv_scores_r2.mean():.3f} ± {cv_scores_r2.std():.3f}")
print(f"RMSE mean ± std: {cv_scores_rmse.mean():.3f} ± {cv_scores_rmse.std():.3f}")

# --------------------------------------------------
# 5. Hold-out test evaluation
# --------------------------------------------------

rf.fit(X_train, y_train)
y_pred_test = rf.predict(X_test)

r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("\n=== Hold-out Test Set Performance ===")
print(f"R2   (test) : {r2_test:.3f}")
print(f"RMSE (test) : {rmse_test:.3f} MPa")

# --------------------------------------------------
# 6. Cross-validated predictions
# --------------------------------------------------

y_pred_cv = cross_val_predict(rf, X, y, cv=cv, n_jobs=-1)
df["strength_pred_cv"] = y_pred_cv

# --------------------------------------------------
# 7. Feature importance visualization
# --------------------------------------------------

importances = rf.feature_importances_
indices_sorted = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 5))
plt.bar(range(len(feature_cols)), importances[indices_sorted])
plt.xticks(range(len(feature_cols)), np.array(feature_cols)[indices_sorted], rotation=45, ha="right")
plt.ylabel("Feature importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 8. Predicted vs Measured plot
# --------------------------------------------------

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_test, alpha=0.7)
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.xlabel("Measured strength (MPa)")
plt.ylabel("Predicted strength (MPa)")
plt.title(f"Measured vs Predicted (R2 = {r2_test:.2f}, RMSE = {rmse_test:.2f} MPa)")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 9. Predicted strength distribution by cooling regime (Boxplot)
# --------------------------------------------------

groups = df.groupby("cooling_regime_label")["strength_pred_cv"].apply(list)
labels = list(groups.index)
data_box = [groups[label] for label in labels]

plt.figure(figsize=(8, 5))
plt.boxplot(data_box, labels=labels, showmeans=True)
plt.ylabel("Predicted compressive strength (MPa)")
plt.title("Predicted strength distribution by cooling regime")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 10. Predicted strength vs age by cooling regime
# --------------------------------------------------

pivot_mean = df.pivot_table(
    index="age_days",
    columns="cooling_regime_label",
    values="strength_pred_cv",
    aggfunc="mean"
).sort_index()

plt.figure(figsize=(8, 5))
for col in pivot_mean.columns:
    plt.plot(pivot_mean.index, pivot_mean[col], marker="o", label=col)

plt.xlabel("Age (days)")
plt.ylabel("Predicted strength (MPa)")
plt.title("Predicted compressive strength vs age for each cooling regime")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
