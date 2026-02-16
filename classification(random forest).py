import pandas as pd 
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv(r"C:\Users\panch\Downloads\weather_classification_data.csv")
X=df.drop("Weather Type",axis=1)
Y=df["Weather Type"]
for cols in X.columns:
    if X[cols].dtype=="object":
        le=LabelEncoder()
        X[cols]=le.fit_transform(X[cols])
le_target=LabelEncoder()
y=le_target.fit_transform(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
classification_model=RandomForestClassifier()
classification_model.fit(X_train,Y_train)
pred=classification_model.predict(X_test)
report=classification_report(Y_test,pred)
print("Accuracy:", accuracy_score(Y_test, pred))
print(report)

