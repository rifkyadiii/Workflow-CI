# 🚀 Churn Prediction Model - MLflow CI/CD Pipeline

Automated machine learning pipeline untuk prediksi customer churn menggunakan MLflow, GitHub Actions, dan Docker deployment.

## 📋 Overview

Pipeline ini secara otomatis:
- ✅ Train RandomForest model untuk prediksi churn
- ✅ Track experiments dengan MLflow
- ✅ Build Docker image dari trained model
- ✅ Push ke Docker Hub untuk deployment

## 🏗️ Architecture

```
GitHub Push → MLflow Training → Model Logging → Docker Build → Docker Hub
```

## 📁 Project Structure

```
.
├── .github/workflows/main.yml    # CI/CD pipeline
├── MLProject/
│   ├── conda.yaml               # Environment dependencies
│   ├── MLProject                # MLflow project config
│   ├── modelling.py            # Model training script
│   └── dataset_preprocessing/
│       └── Telco-Customer-Churn_preprocessing.csv
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- GitHub repository dengan secrets:
  - `DOCKERHUB_USERNAME`
  - `DOCKERHUB_TOKEN`

### Deployment
1. Push ke `main` branch
2. GitHub Actions otomatis menjalankan pipeline
3. Docker image tersedia di Docker Hub: `your-username/churn-prediction-api`

### Running the Model
```bash
# Pull dan jalankan Docker image
docker pull your-username/churn-prediction-api:latest
docker run -p 8080:8080 your-username/churn-prediction-api:latest

# Test API endpoint
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": [...], "data": [[...]]}}'
```

## 🔧 Model Details

- **Algorithm**: Random Forest Classifier
- **Framework**: Scikit-learn
- **Features**: Customer demographics, services, account info
- **Target**: Churn (Yes/No)
- **Accuracy**: ~80%+ (varies per run)

## 📊 MLflow Tracking

Model experiments dan metrics otomatis di-track di MLflow:
- Parameters: `n_estimators`, `max_depth`
- Metrics: `accuracy`, `train_samples`, `test_samples`
- Artifacts: Model binary, signature

## 🐳 Docker Image

Image berisi:
- Python 3.10 runtime
- MLflow model server
- Pre-trained churn prediction model
- REST API endpoint di port 8080

## 🔄 CI/CD Pipeline

Workflow otomatis trigger saat push ke `main`:

1. **Setup Environment** - Install MLflow & dependencies
2. **Create Conda Env** - Setup training environment
3. **Train Model** - Run MLflow project
4. **Extract Run ID** - Parse MLflow output
5. **Build Docker** - Create containerized API
6. **Push to Hub** - Deploy ke Docker Hub

## 📝 Configuration

### Environment Dependencies (`conda.yaml`)
```yaml
dependencies:
  - python=3.10
  - pip:
    - mlflow==2.13.0
    - scikit-learn==1.3.0
    - pandas==2.0.3
```

### Model Parameters
- `n_estimators`: 100
- `max_depth`: 20
- `test_size`: 20%
- `random_state`: 42

## 🛠️ Local Development

```bash
# Setup environment
conda env create -f MLProject/conda.yaml
conda activate churn-env

# Run training
mlflow run ./MLProject

# Serve model locally
mlflow models serve -m runs:/<RUN_ID>/model -p 8080
```

## 📈 Monitoring

- **GitHub Actions**: Pipeline status & logs
- **Docker Hub**: Image versions & pull stats
- **MLflow**: Experiment tracking & model registry

## 🔗 Links

- **Docker Hub**: `https://hub.docker.com/r/your-username/churn-prediction-api`
- **MLflow UI**: Available saat local development
- **GitHub Actions**: Tab "Actions" di repository

## 📄 License

MIT License - Feel free to use dan modify!

---

> **Note**: Replace `your-username` dengan actual Docker Hub username Anda di secrets dan documentation.
