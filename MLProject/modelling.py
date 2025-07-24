import pandas as pd
import mlflow
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("=" * 50)
print("STARTING MLFLOW CHURN PREDICTION MODEL TRAINING")
print("=" * 50)

with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"MLflow run started with ID: {run_id}")
    print(f"Started run with ID {run_id}")  # Format yang lebih mudah di-parse
    
    # Muat Data
    data_path = 'dataset_preprocessing/Telco-Customer-Churn_preprocessing.csv'
    print(f"Loading data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    
    X = df.drop('Churn', axis=1)  
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Latih model dengan parameter tetap
    n_estimators = 100
    max_depth = 20
    
    print(f"Training model with n_estimators={n_estimators}, max_depth={max_depth}")
    
    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluasi dan Log Metrik
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model trained with accuracy: {accuracy:.4f}")
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("train_samples", len(X_train))
    mlflow.log_metric("test_samples", len(X_test))
    
    # Log model dengan signature untuk better compatibility
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature
    
    signature = infer_signature(X_train, y_pred)
    
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name="churn-prediction-model"
    )
    
    print(f"Model logged successfully with run ID: {run_id}")
    print(f"Run completed successfully!")
    print(f"FINAL RUN ID: {run_id}")
    print("=" * 50)