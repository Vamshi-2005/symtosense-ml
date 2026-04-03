import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# load dataset
data = pd.read_csv("disease_dataset.csv")

# split features and label
X = data.drop("disease", axis=1)
y = data["disease"]

# create model
model = DecisionTreeClassifier()

# train model
model.fit(X, y)

# save trained model
joblib.dump(model, "disease_model.pkl")

print("Model trained successfully and saved as disease_model.pkl")