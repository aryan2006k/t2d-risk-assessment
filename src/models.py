# src/models.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report)
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class T2DModelTrainer:
    """Train and evaluate multiple models for T2D risk assessment"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.model_scores = {}
        
    def get_base_models(self):
        """Initialize base models with interpretability in mind"""
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=42, max_iter=1000, class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=42, max_depth=10,
                min_samples_split=10, class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=42, max_depth=5,
                learning_rate=0.1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100, random_state=42, max_depth=5,
                learning_rate=0.1, use_label_encoder=False,
                eval_metric='logloss'
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100, random_state=42, max_depth=5,
                learning_rate=0.1, verbose=-1
            )
        }
        return models
    
    def train_model(self, model, X_train, y_train, X_val, y_val):
        """Train a single model"""
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_val_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred, y_train_proba)
        val_metrics = self.calculate_metrics(y_val, y_val_pred, y_val_proba)
        
        return model, train_metrics, val_metrics
    
    def calculate_metrics(self, y_true, y_pred, y_proba):
        """Calculate comprehensive metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        return metrics
    
    def hyperparameter_tuning(self, model_name, model, X_train, y_train, cv=5):
        """Perform hyperparameter tuning"""
        param_grids = {
            'Logistic Regression': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [5, 10, 20]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3]
            }
        }
        
        if model_name in param_grids:
            grid_search = GridSearchCV(
                model, param_grids[model_name], 
                cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_
        
        return model
    
    def train_all_models(self, X_train, y_train, X_val, y_val, tune_hyperparameters=False):
        """Train all models and select the best one"""
        base_models = self.get_base_models()
        
        print("Training models...")
        for name, model in base_models.items():
            print(f"\nTraining {name}...")
            
            # Optionally tune hyperparameters
            if tune_hyperparameters and name in ['Logistic Regression', 'Random Forest', 'XGBoost']:
                print(f"  Tuning hyperparameters...")
                model = self.hyperparameter_tuning(name, model, X_train, y_train)
            
            # Train model
            trained_model, train_metrics, val_metrics = self.train_model(
                model, X_train, y_train, X_val, y_val
            )
            
            # Store model and scores
            self.models[name] = trained_model
            self.model_scores[name] = {
                'train': train_metrics,
                'validation': val_metrics
            }
            
            # Print results
            print(f"  Train ROC-AUC: {train_metrics['roc_auc']:.4f}")
            print(f"  Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Val F1-Score: {val_metrics['f1_score']:.4f}")
        
        # Select best model based on validation ROC-AUC
        best_score = 0
        for name, scores in self.model_scores.items():
            val_auc = scores['validation']['roc_auc']
            if val_auc > best_score:
                best_score = val_auc
                self.best_model = self.models[name]
                self.best_model_name = name
        
        print(f"\n{'='*50}")
        print(f"Best Model: {self.best_model_name}")
        print(f"Validation ROC-AUC: {best_score:.4f}")
        
        return self.best_model
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model on test set"""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = self.calculate_metrics(y_test, y_pred, y_proba)
        
        print("\nTest Set Performance:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  {metrics['confusion_matrix']}")
        
        return metrics
    
    def save_model(self, model, filepath):
        """Save trained model"""
        joblib.dump(model, filepath)
        
    def load_model(self, filepath):
        """Load saved model"""
        return joblib.load(filepath)