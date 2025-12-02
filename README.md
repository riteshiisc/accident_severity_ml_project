# Accident Severity Prediction — README

## Project Title

**Accident Severity Prediction Using UK Road Safety Data**

## Team Members (Team–21 Data Warriors)

| Name | Email |
|------|-------|
| Anil Chandra | anilchandrad@iisc.ac.in |
| Gomathi Sankar S | gomathis@iisc.ac.in |
| Neeraj Kumar | neeraj3@iisc.ac.in |
| Ritesh Mishra | riteshmishra@iisc.ac.in |

## Problem Statement

Road accidents remain a major public safety issue despite improvements in technology and infrastructure. Accident severity is influenced by a complex interaction of factors such as weather, lighting, road type, and driver behavior.

**Goal**: Predict accident severity to support proactive planning, risk reduction, and faster emergency response.

## Dataset Description

The project uses UK government road safety datasets published by the Department for Transport and hosted on Kaggle:

### 1. Accident_Information.csv
- **One record = one accident**
- **Date Range**: 2005–2017
- **Includes ~34 features** such as:
  - `Accident_Index` (Primary Key)
  - Location (Latitude/Longitude, OSGR coordinates)
  - Time & Date
  - Weather conditions
  - Road type, speed limit
  - Number of vehicles, casualties
  - Accident severity

### 2. Vehicle_Information.csv
- **One record = one vehicle involved in an accident**
- **Date Range**: 2004–2016
- **~24+ vehicle-level features**:
  - Vehicle type
  - Driver characteristics
  - Vehicle condition
  - Manoeuvres, impact point
  - Engine capacity, vehicle age

### Dataset Summary
- **Datasets linked via**: `Accident_Index`
- **Total combined features**: ~57
- **Record count**: 2M accidents + 1.5M vehicle entries

**Dataset Source**: [UK Road Safety Accidents and Vehicles](https://www.kaggle.com/datasets/tsiaras/uk-road-safety-accidents-and-vehicles/data)

## High-Level Approach

### 1. Data Layer
- Load CSVs
- Merge Accident & Vehicle datasets
- Handle missing values, skew, noise

### 2. Exploratory Data Analysis (EDA)
- Univariate, bivariate & multivariate analysis
- Visual trends across weather, light, seasons, road type
- Spatial plots across UK

### 3. Feature Engineering
- Derived features (Hour, Month, Is_Weekend, vehicle averages)
- Encoding (Ordinal or native categorical encoders)
- Handling class imbalance (Stratified sampling, class weights)

### 4. Modeling
Tree-based classification models:
- Random Forest
- XGBoost
- LightGBM
- CatBoost

### 5. Evaluation
**Metrics**:
- Accuracy
- Macro F1 (critical for Serious & Fatal classes)
- Weighted F1
- Confusion Matrix
- SHAP interpretability

## Summary of Results

### Model Performance Highlights

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| Random Forest | 0.846 | 0.308 | 0.777 |
| XGBoost | 0.834 | 0.322 | 0.782 |
| **LightGBM (Best)** | **0.6159** | **0.379** | **0.6798** |
| CatBoost | 0.847 | 0.313 | 0.780 |

### Why LightGBM is Best
- **Highest Macro F1 = 0.379**, balanced across classes
- Better captures Serious and Fatal patterns
- Most reliable for real-world safety applications

### Key Predictive Drivers (from SHAP)
- Speed limit
- Number of vehicles
- Junction details & control
- Engine capacity & vehicle age
- Weather & light conditions
- Urban/rural location

## Final Outcome

The project successfully built an end-to-end ML pipeline that:
- Processes and analyzes 3.5M+ accident & vehicle records
- Predicts severity with strong class-balanced performance
- Identifies critical safety risk factors

## Key Insights

### Target Variable Distribution
- **Slight accidents**: 87.1% (majority class)
- **Serious accidents**: 12.8% (moderate representation)
- **Fatal accidents**: 0.1% (extremely rare)

### Class Imbalance Challenges
- Severe imbalance requires specialized ML techniques
- Standard accuracy metrics can be misleading
- Focus on Macro F1-score for balanced evaluation

### Business Impact
- **30% Fatal Reduction** target
- **25% Serious Injury Reduction** target
- Improved emergency response optimization
- Evidence-based safety policy development

## License

This project is for educational purposes as part of the Data Science in Practice course at IISc.

## Contributing

This is a team project for academic purposes. For questions or collaboration, please contact the team members listed above.

---

**Data Science in Practice Project | December 2025 | Indian Institute of Science**
