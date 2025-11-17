
"""
Task 3: Predictive Analytics for Resource Allocation
Random Forest Classifier + Confusion Matrix + Feature Importance Plot
Designed for VS Code
"""

# -----------------------------
# 📌 Import Libraries
# -----------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# -----------------------------
# 📌 Step 1: Load Dataset
# -----------------------------
try:
    data = pd.read_csv("breast_cancer_data.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ ERROR: File 'breast_cancer_data.csv' not found.")
    exit()

print("\nDataset Preview:")
print(data.head())

# -----------------------------
# 📌 Step 2: Encode Labels
# -----------------------------
label_encoder = LabelEncoder()
data["diagnosis"] = label_encoder.fit_transform(data["diagnosis"])
# 0 = benign, 1 = malignant
print("\nLabel encoding completed.")

# -----------------------------
# 📌 Step 3: Feature Split
# -----------------------------
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nDataset successfully split into train/test.")

# -----------------------------
# 📌 Step 4: Train Random Forest
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
print("\nModel training completed.")

# -----------------------------
# 📌 Step 5: Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 📌 Step 6: Evaluation Metrics
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n📊 Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

# =============================
# 📊 GRAPH 1 — CONFUSION MATRIX
# =============================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# ====================================
# 📊 GRAPH 2 — FEATURE IMPORTANCE PLOT
# ====================================
importances = model.feature_importances_
feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_df["Feature"], feature_df["Importance"])
plt.gca().invert_yaxis()  # Largest importance at top
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature Name")
plt.tight_layout()
plt.show()
