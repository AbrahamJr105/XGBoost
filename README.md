# 🌾 Agricultural Yield Prediction using XGBoost

A machine learning project that predicts agricultural yields based on weather and environmental data using XGBoost regression. This project demonstrates the application of gradient boosting algorithms for agricultural forecasting, which is crucial for food security and agricultural planning.

## 📊 Project Overview

This project implements a complete machine learning pipeline for predicting agricultural yields using weather data. It features:

- **Data-driven approach**: Uses real-world weather and agricultural data
- **XGBoost implementation**: Leverages gradient boosting for accurate predictions
- **Model persistence**: Save and load trained models for future predictions
- **Performance monitoring**: Comprehensive evaluation metrics and resource monitoring
- **Production-ready**: Command-line interface with configurable parameters

## 🚀 Features

- **Training Pipeline** (`XGBoost.py`):
  - Automated data preprocessing and feature selection
  - Hyperparameter configuration with early stopping
  - Model evaluation with multiple metrics (RMSE, MAE, R², MSE)
  - Feature importance visualization
  - Resource usage monitoring (memory, CPU, training time)
  - Optional grid search for hyperparameter optimization

- **Prediction Pipeline** (`predict_xgboost.py`):
  - Load pre-trained models for inference
  - Batch prediction on new datasets
  - Feature validation and reordering
  - Export predictions to CSV

## 📋 Requirements

```
pandas
scikit-learn
xgboost
matplotlib
psutil
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🔧 Usage

### Training a Model

Basic training with default parameters:
```bash
python XGBoost.py --data weather_agriculture_combined_cleand\ \(1\).csv
```

Advanced training with custom parameters:
```bash
python XGBoost.py --data your_dataset.csv --test_size 0.25 --early_stop 15
```

### Making Predictions

```bash
python predict_xgboost.py --model xgboost_model.json --data new_data.csv --output predictions.csv
```

## 📈 Model Performance

The model provides comprehensive performance metrics:

- **RMSE**: Root Mean Square Error for prediction accuracy
- **MAE**: Mean Absolute Error for average prediction deviation
- **R²**: Coefficient of determination for explained variance
- **MSE**: Mean Square Error for loss quantification
- **Inference Time**: Speed of model predictions

## 🗂️ Project Structure

```
XGBoost/
├── XGBoost.py                           # Main training script
├── predict_xgboost.py                   # Prediction script
├── weather_agriculture_combined_cleand (1).csv  # Dataset (~276KB)
├── xgboost_model.json                   # Trained model (saved after training)
├── requirements.txt                     # Python dependencies
└── README.md                           # Project documentation
```

## 📊 Dataset

The project uses a cleaned weather and agriculture combined dataset containing:
- **Size**: ~276KB of processed data
- **Source**: Official weather and agricultural sources
- **Features**: Numeric weather and environmental variables
- **Target**: Agricultural yield values (`value` column)

## 🛠️ Technical Implementation

### Key Features:
- **Gradient Boosting**: XGBoost regression with optimized parameters
- **Early Stopping**: Prevents overfitting during training
- **Feature Selection**: Automatic numeric feature detection
- **Model Persistence**: JSON format for cross-platform compatibility
- **Resource Monitoring**: Track memory usage and training time
- **Visualization**: Feature importance plots for model interpretation

### Hyperparameters:
- Learning Rate: 0.1
- Max Depth: 6
- Subsample: 0.8
- Column Sample by Tree: 0.8
- Objective: Squared error regression

## 🎯 Applications

This project demonstrates practical machine learning applications in:

- **Agricultural Planning**: Yield forecasting for crop management
- **Food Security**: Predicting harvest outcomes for supply chain planning
- **Risk Assessment**: Understanding weather impact on agricultural production
- **Decision Support**: Data-driven insights for farmers and policymakers

## 🔍 Model Insights

The trained model provides:
- Feature importance rankings to identify key weather factors
- Quantitative performance metrics for model reliability
- Resource usage statistics for deployment planning
- Visualization tools for model interpretation

## 🚀 Future Enhancements

Potential improvements and extensions:
- Multi-target prediction for different crop types
- Time series forecasting with seasonal patterns
- Integration with real-time weather APIs
- Web interface for interactive predictions
- Ensemble methods combining multiple algorithms

## 📝 License

This project is open source and available for educational and research purposes.

---

*This project showcases practical machine learning implementation for agricultural prediction, demonstrating proficiency in data science, model development, and production-ready code structure.*
