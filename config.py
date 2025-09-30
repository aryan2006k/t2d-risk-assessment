# config.py
import os

class Config:
    """Configuration for T2D Risk Assessment Framework"""
    
    # Model Parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    
    # Feature Categories
    DEMOGRAPHIC_FEATURES = ['Age', 'Gender', 'BMI', 'Ethnicity']
    CLINICAL_FEATURES = ['Glucose', 'HbA1c', 'BloodPressure', 'Cholesterol', 
                        'HDL', 'LDL', 'Triglycerides']
    LIFESTYLE_FEATURES = ['PhysicalActivity', 'DietQuality', 'SmokingStatus', 
                         'AlcoholConsumption', 'SleepHours', 'StressLevel']
    FAMILY_HISTORY = ['FamilyHistoryDiabetes', 'FamilyHistoryHypertension']
    COMORBIDITIES = ['Hypertension', 'Obesity', 'PCOS', 'GestationalDiabetes']
    
    # Risk Thresholds
    LOW_RISK_THRESHOLD = 0.3
    MODERATE_RISK_THRESHOLD = 0.6
    HIGH_RISK_THRESHOLD = 0.8
    
    # Model Paths
    MODEL_PATH = "models/"
    DATA_PATH = "data/"
    OUTPUT_PATH = "outputs/"
    
    # Hugging Face
    HF_SPACE_NAME = "t2d-risk-assessment"
    HF_USERNAME = "your-username"  # Replace with your username

    # Interpretability
    N_SAMPLES_LIME = 1000
    N_SAMPLES_SHAP = 100
    