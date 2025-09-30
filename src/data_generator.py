# src/data_generator.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

class T2DDataGenerator:
    """Generate synthetic EHR data for T2D risk assessment"""
    
    def __init__(self, n_samples=10000, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_demographic_data(self):
        """Generate demographic features"""
        data = {
            'PatientID': [f'P{i:05d}' for i in range(self.n_samples)],
            'Age': np.random.normal(45, 15, self.n_samples).clip(18, 85).astype(int),
            'Gender': np.random.choice(['Male', 'Female'], self.n_samples, p=[0.48, 0.52]),
            'Ethnicity': np.random.choice(
                ['Caucasian', 'African American', 'Hispanic', 'Asian', 'Other'],
                self.n_samples, p=[0.4, 0.2, 0.2, 0.15, 0.05]
            ),
            'BMI': np.random.normal(27, 5, self.n_samples).clip(16, 45)
        }
        return pd.DataFrame(data)
    
    def generate_clinical_data(self, demographics_df):
        """Generate clinical measurements based on demographics"""
        clinical_data = {}
        
        # Correlate clinical values with age and BMI
        age_factor = (demographics_df['Age'] - 18) / 67  # Normalize age
        bmi_factor = (demographics_df['BMI'] - 16) / 29  # Normalize BMI
        
        # Glucose levels (mg/dL) - higher for older and higher BMI
        base_glucose = 90 + 30 * age_factor + 20 * bmi_factor
        clinical_data['Glucose'] = np.random.normal(base_glucose, 15).clip(70, 200)
        
        # HbA1c (%) - correlated with glucose
        clinical_data['HbA1c'] = 4.0 + (clinical_data['Glucose'] - 70) * 0.02 + \
                                 np.random.normal(0, 0.3, self.n_samples)
        clinical_data['HbA1c'] = clinical_data['HbA1c'].clip(4.0, 10.0)
        
        # Blood Pressure
        systolic = 110 + 20 * age_factor + 15 * bmi_factor + \
                  np.random.normal(0, 10, self.n_samples)
        clinical_data['SystolicBP'] = systolic.clip(90, 180)
        clinical_data['DiastolicBP'] = (systolic * 0.6 + \
                                       np.random.normal(0, 5, self.n_samples)).clip(60, 110)
        
        # Cholesterol levels
        clinical_data['TotalCholesterol'] = np.random.normal(200, 40, self.n_samples).clip(120, 350)
        clinical_data['HDL'] = np.random.normal(50 - 10 * bmi_factor, 15).clip(20, 100)
        clinical_data['LDL'] = clinical_data['TotalCholesterol'] * 0.6 + \
                               np.random.normal(0, 20, self.n_samples)
        clinical_data['Triglycerides'] = np.random.normal(150 + 50 * bmi_factor, 50).clip(50, 500)
        
        return pd.DataFrame(clinical_data)
    
    def generate_lifestyle_data(self, demographics_df):
        """Generate lifestyle factors"""
        lifestyle_data = {}
        
        # Physical Activity (hours per week)
        age_factor = (demographics_df['Age'] - 18) / 67
        lifestyle_data['PhysicalActivity'] = np.random.exponential(3, self.n_samples) * \
                                            (1 - 0.5 * age_factor)
        lifestyle_data['PhysicalActivity'] = lifestyle_data['PhysicalActivity'].clip(0, 20)
        
        # Diet Quality Score (0-10)
        lifestyle_data['DietQuality'] = np.random.beta(3, 3, self.n_samples) * 10
        
        # Smoking Status
        lifestyle_data['SmokingStatus'] = np.random.choice(
            ['Never', 'Former', 'Current'], 
            self.n_samples, p=[0.5, 0.3, 0.2]
        )
        
        # Alcohol Consumption (drinks per week)
        lifestyle_data['AlcoholConsumption'] = np.random.exponential(2, self.n_samples).clip(0, 20)
        
        # Sleep Hours
        lifestyle_data['SleepHours'] = np.random.normal(7, 1.5, self.n_samples).clip(4, 12)
        
        # Stress Level (0-10)
        lifestyle_data['StressLevel'] = np.random.beta(2, 3, self.n_samples) * 10
        
        return pd.DataFrame(lifestyle_data)
    
    def generate_medical_history(self):
        """Generate family history and comorbidities"""
        history_data = {
            'FamilyHistoryDiabetes': np.random.choice([0, 1], self.n_samples, p=[0.7, 0.3]),
            'FamilyHistoryHypertension': np.random.choice([0, 1], self.n_samples, p=[0.6, 0.4]),
            'Hypertension': np.random.choice([0, 1], self.n_samples, p=[0.75, 0.25]),
            'Obesity': np.random.choice([0, 1], self.n_samples, p=[0.7, 0.3]),
            'PCOS': np.random.choice([0, 1], self.n_samples, p=[0.95, 0.05]),
            'GestationalDiabetes': np.random.choice([0, 1], self.n_samples, p=[0.97, 0.03])
        }
        return pd.DataFrame(history_data)
    
    def generate_target(self, full_df):
        """Generate T2D risk based on features"""
        risk_score = 0
        
        # Age contribution
        risk_score += (full_df['Age'] > 45).astype(int) * 0.15
        risk_score += (full_df['Age'] > 65).astype(int) * 0.1
        
        # BMI contribution
        risk_score += (full_df['BMI'] > 25).astype(int) * 0.1
        risk_score += (full_df['BMI'] > 30).astype(int) * 0.15
        
        # Clinical indicators
        risk_score += (full_df['Glucose'] > 100).astype(int) * 0.2
        risk_score += (full_df['HbA1c'] > 5.7).astype(int) * 0.25
        risk_score += (full_df['SystolicBP'] > 130).astype(int) * 0.05
        
        # Lifestyle factors
        risk_score += (full_df['PhysicalActivity'] < 2.5).astype(int) * 0.1
        risk_score += (full_df['DietQuality'] < 5).astype(int) * 0.1
        risk_score += (full_df['SmokingStatus'] == 'Current').astype(int) * 0.1
        
        # Family history and comorbidities
        risk_score += full_df['FamilyHistoryDiabetes'] * 0.15
        risk_score += full_df['Hypertension'] * 0.1
        risk_score += full_df['Obesity'] * 0.1
        
        # Add noise
        risk_score += np.random.normal(0, 0.1, self.n_samples)
        
        # Convert to binary outcome (1 = High Risk/T2D)
        threshold = 0.5
        return (risk_score > threshold).astype(int)
    
    def generate_dataset(self):
        """Generate complete synthetic dataset"""
        # Generate all components
        demographics = self.generate_demographic_data()
        clinical = self.generate_clinical_data(demographics)
        lifestyle = self.generate_lifestyle_data(demographics)
        history = self.generate_medical_history()
        
        # Combine all features
        full_df = pd.concat([demographics, clinical, lifestyle, history], axis=1)
        
        # Generate target
        full_df['T2D_Risk'] = self.generate_target(full_df)
        
        # Add timestamp
        full_df['RecordDate'] = pd.date_range(
            start=datetime.now() - timedelta(days=365),
            periods=self.n_samples,
            freq='H'
        )[:self.n_samples]
        
        return full_df

# Generate and save dataset
if __name__ == "__main__":
    generator = T2DDataGenerator(n_samples=10000)
    dataset = generator.generate_dataset()
    dataset.to_csv('t2d_synthetic_data.csv', index=False)
    print(f"Generated dataset with {len(dataset)} samples")
    print(f"Class distribution:\n{dataset['T2D_Risk'].value_counts()}")