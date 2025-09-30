# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib

class T2DPreprocessor:
    """Preprocessing pipeline for T2D risk assessment"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.imputer = None
        
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Impute numeric columns with median
        if len(numeric_cols) > 0:
            numeric_imputer = SimpleImputer(strategy='median')
            df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
        
        # Impute categorical columns with mode
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_value, inplace=True)
        
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical variables"""
        categorical_cols = ['Gender', 'Ethnicity', 'SmokingStatus']
        encoded_dfs = []
        
        for col in categorical_cols:
            if col in df.columns:
                # One-hot encoding
                encoder = pd.get_dummies(df[col], prefix=col, drop_first=False)
                encoded_dfs.append(encoder)
                self.encoders[col] = df[col].unique()
        
        # Drop original categorical columns
        df = df.drop(columns=categorical_cols, errors='ignore')
        
        # Concatenate encoded columns
        if encoded_dfs:
            df = pd.concat([df] + encoded_dfs, axis=1)
        
        return df
    
    def scale_features(self, X_train, X_val=None, X_test=None):
        """Scale numerical features"""
        scaler = StandardScaler()
        
        # Identify numerical columns
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns
        
        # Fit and transform training data
        X_train_scaled = X_train.copy()
        X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        
        self.scalers['standard'] = scaler
        
        # Transform validation and test data if provided
        X_val_scaled = None
        X_test_scaled = None
        
        if X_val is not None:
            X_val_scaled = X_val.copy()
            X_val_scaled[numerical_cols] = scaler.transform(X_val[numerical_cols])
        
        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def create_feature_interactions(self, df):
        """Create interaction features for better predictions"""
        interactions = {}
        
        if 'BMI' in df.columns and 'Age' in df.columns:
            interactions['BMI_Age'] = df['BMI'] * df['Age'] / 100
        
        if 'Glucose' in df.columns and 'HbA1c' in df.columns:
            interactions['Glucose_HbA1c'] = df['Glucose'] * df['HbA1c'] / 100
        
        if 'PhysicalActivity' in df.columns and 'BMI' in df.columns:
            interactions['Activity_BMI_Ratio'] = df['PhysicalActivity'] / (df['BMI'] + 1)
        
        if 'SystolicBP' in df.columns and 'DiastolicBP' in df.columns:
            interactions['PulsePressure'] = df['SystolicBP'] - df['DiastolicBP']
            interactions['MeanArterialPressure'] = (df['SystolicBP'] + 2 * df['DiastolicBP']) / 3
        
        # Add interaction features to dataframe
        for name, values in interactions.items():
            df[name] = values
        
        return df
    
    def balance_dataset(self, X, y, random_state=42):
        """Balance dataset using SMOTE"""
        smote = SMOTE(random_state=random_state)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        return X_balanced, y_balanced
    
    def preprocess(self, df, target_col='T2D_Risk', test_size=0.2, val_size=0.1):
        """Complete preprocessing pipeline"""
        # Drop non-feature columns
        drop_cols = ['PatientID', 'RecordDate'] if 'PatientID' in df.columns else []
        df = df.drop(columns=drop_cols, errors='ignore')
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Create interaction features
        df = self.create_feature_interactions(df)
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), 
            random_state=42, stratify=y_temp
        )
        
        # Scale features
        X_train, X_val, X_test = self.scale_features(X_train, X_val, X_test)
        
        # Balance training data
        X_train, y_train = self.balance_dataset(X_train, y_train)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessor(self, filepath):
        """Save preprocessor for later use"""
        joblib.dump(self, filepath)
    
    @staticmethod
    def load_preprocessor(filepath):
        """Load saved preprocessor"""
        return joblib.load(filepath)