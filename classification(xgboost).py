import pandas as pd 
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
df = pd.read_csv(r"C:\Users\panch\Downloads\weather_classification_data.csv")
X = df.drop("Weather Type", axis=1)
Y = df["Weather Type"]
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
le_target = LabelEncoder()
y = le_target.fit_transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
model.fit(X_train, Y_train)
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, pred))
print(classification_report(Y_test, pred))