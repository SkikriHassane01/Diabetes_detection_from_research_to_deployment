# End to End Diabetes Classification

## **Project Overview**

This Project aims to develop a machine learning model to classify patients into three categories:

- `No Diabetes (0)`
- `Prediabetes (1)`
- `Diabetes (2)`

The Model will achieve at least **+90% accuracy and recall**, wi will focus on the recall metric because the False negative (missed diagnoses, the patient is defected with the diabetes and the model predict that he is not) are more dangerous than false positive (can ruled out with additional tests). This classification model will eventually be integrated into a larger healthcare application that identifies multiple diseases.

## **Project Objectives**

- Develop a robust diabetes classification model with > 90% accuracy and recall
- Address class imbalance
- Identify the most effective modeling technique through comparative analysis
- Optimize hyperparameter using ***Optuna***
- Implement MlOps best practices with ***ZenMl*** and ***MLflow***
- Create scalable, maintainable code using design patterns
  - ***Scalable code***: Code that can grow without major rewrites.
  - ***Maintainable code***: code that's easy to understand, modify, and debug (clear organization, will-documented, and following consistent standards)

## **Dataset Understanding**

The dataset includes:

### Target Variable

- **Diabetes_012**: Diabetes classification (0: No diabetes, 1: Prediabetes, 2: Diabetes)

### Features

**Binary Features (0/1)**:

- Health Indicators: HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, etc.
- Lifestyle: Fruits, Veggies, HvyAlcoholConsump
- Healthcare Access: AnyHealthcare, NoDocbcCost
- Physical Condition: DiffWalk
- Demographics: Sex

**Continuous/Ordinal Features**:

- BMI: Body Mass Index
- GenHlth: General Health (1-5 scale)
- MentHlth: Mental Health Days
- PhysHlth: Physical Health Days
- Age: Age Category (1-13)
- Education: Education Level

## **Data Pipeline Architecture**

### ***1. Data Ingestion, validation, versioning***

- Load raw data
- Initial validation checks
- Data versioning with MlFlow

### ***2. Exploratory Data Analysis (EDA)***

- Distribution analysis for all features
- Target class imbalance assessment
- Correlation analysis with target variable
- Visualization of key relationships
- Feature importance with RandomForest Classifier

### ***3. Data Processing***

- Handling missing values
- Handling the duplicated values
- Outlier detection and treatment
- Feature scaling for continuous variables
- Data type validation and conversion

### ***4. Feature Engineering***

- Create new Features like `HealthScoreCalculation` or `RiskScoreCalculation`...

### ***5. Class Imbalance Handling***

- Implementation of `SMOTE` for  synthetic data generation
- Evaluation of different SMOTE variants (Borderline-SMOTE, ADASYN)

## **Modeling Strategy**

### ***1. Baseline Model Evaluation***

Evaluate multiple classification algorithm:

- ***Logistic Regression***
- ***Random Forest***
- ***XGBoost***
- ***LightGBM***
- ***CatBoost***
- ***KNN***
- ***Neural Networks***

### ***2. Model Selection Criteria***

- Primary metrics: Accuracy and Recall (target >90%)
- Secondary metrics: Precision, F1-Score, AUC-PR
- Cross-validation using stratified k-fold (k=5)

### ***3. Hyperparameter Optimization***

- Implement Optuna for the best-performing model
- Define optimization objectives (maximize recall while maintaining accuracy)
- Optimization search space based on model type
- Utilize Bayesian optimization approach
- Cross-validation within optimization loop

## **MLOps Implementation**

### ***1. Experiment Tracking with MLflow***

- Track all experiments across different models
- Log hyperparameter, metrics, and artifacts
- Version models and datasets
- Compare experiment results

### ***2. Pipeline Orchestration with ZenML***

- Create reproducible pipeline steps:
  - Data ingestion
  - Preprocessing
  - Feature engineering
  - Model training
  - Model evaluation
  - Model deployment
- Configure caching for efficient iteration
- Implement pipeline versioning

### ***3. Continuous Integration/Continuous Deployment `(CI/CD)`***

This setup ensures reliable, production-ready ML models with automated quality checks and deployment processes.

- Automated testing framework
- Model validation before deployment
- Continuous model retraining triggers
- Model serving infrastructure

### ***4. Monitoring and Maintenance***

- Data drift detection
- Concept drift monitoring
- Performance metric tracking
- Automated retraining based on drift thresholds
- Alert system for performance degradation

## **Technical Stack used in this project**

### **Core Technologies**

- **Programming Language**: Python 3.8+
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM, catboost
- **Visualization**: Matplotlib, Seaborn
- **MLOps**: ZenML, MLflow

### Development Tools

- **Version Control**: Git, Github
- **Virtual Environment**: Conda ...
- **Testing**: pytest
- **Code Quality**: Black, Flake8, mypy