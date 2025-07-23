import pandas as pd
import mlflow
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set tracking URI dengan path absolut
workspace_path = os.environ.get('GITHUB_WORKSPACE', '/home/runner/work/Workflow-CI/Workflow-CI')
mlflow.set_tracking_uri(f"file://{workspace_path}/mlruns")

# Muat Data
df = pd.read_csv('dataset_preprocessing/Telco-Customer-Churn_preprocessing.csv')
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model dengan parameter terbaik
n_estimators = 100
max_depth = 20
mlflow.log_param("n_estimators", n_estimators)
mlflow.log_param("max_depth", max_depth)
model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
model.fit(X_train, y_train)

# Evaluasi dan Log
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
mlflow.log_metric("accuracy", accuracy)

# Log model
mlflow.sklearn.log_model(model, "model")