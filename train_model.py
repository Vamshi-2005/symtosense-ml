import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# 🔥 LOAD DATA
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "dataset.csv"))
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"📊 Diseases: {df['disease'].nunique()} unique")

# 🔥 FEATURES + LABEL
X = df.drop("disease", axis=1)
y = df["disease"]
cols = list(X.columns)

# 🔥 TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🔥 TRAIN MODEL
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# 🔥 EVALUATE
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy * 100:.1f}%")

# 🔥 SAVE MODEL + COLUMNS
joblib.dump(model, os.path.join(BASE_DIR, "model.pkl"))
joblib.dump(cols, os.path.join(BASE_DIR, "columns.pkl"))
print(f"\n✅ Model saved: model.pkl")
print(f"✅ Columns saved: columns.pkl ({len(cols)} symptoms)")