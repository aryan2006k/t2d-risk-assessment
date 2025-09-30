# train.py
import pandas as pd
import numpy as np
import joblib
from src.data_generator import T2DDataGenerator
from src.preprocessing import T2DPreprocessor
from src.models import T2DModelTrainer
from src.interpretability import T2DInterpreter
import os

def main():
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # Step 1: Generate or load data
    print("Generating synthetic dataset...")
    generator = T2DDataGenerator(n_samples=10000)
    dataset = generator.generate_dataset()
    dataset.to_csv('data/t2d_dataset.csv', index=False)
    print(f"Dataset shape: {dataset.shape}")
    
    # Step 2: Preprocess data
    print("\nPreprocessing data...")
    preprocessor = T2DPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess(
        dataset, target_col='T2D_Risk'
    )
    
    # Save preprocessor
    preprocessor.save_preprocessor('models/preprocessor.pkl')
    
    # Step 3: Train models
    print("\nTraining models...")
    trainer = T2DModelTrainer()
    best_model = trainer.train_all_models(
        X_train, y_train, X_val, y_val, 
        tune_hyperparameters=True
    )
    
    # Step 4: Evaluate on test set
    print("\n" + "="*50)
    print("FINAL TEST SET EVALUATION")
    print("="*50)
    test_metrics = trainer.evaluate_model(best_model, X_test, y_test)
    
    # Step 5: Save best model
    trainer.save_model(best_model, 'models/best_model.pkl')
    joblib.dump(preprocessor.feature_names, 'models/feature_names.pkl')
    
    # Step 6: Test interpretability
    print("\nTesting interpretability module...")
    interpreter = T2DInterpreter(
        best_model, 
        preprocessor.feature_names,
        X_train
    )
    
    # Explain a sample prediction
    sample_idx = 0
    X_sample = X_test.iloc[[sample_idx]]
    explanation = interpreter.explain_prediction(X_sample, method='shap')
    
    print("\nSample Prediction Explanation:")
    print(f"Risk Score: {explanation['prediction']:.2%}")
    print("Top Contributing Factors:")
    for i, (feature, impact) in enumerate(list(explanation['feature_impacts'].items())[:5], 1):
        print(f"  {i}. {feature}: {impact:.3f}")
    
    print("\nTraining completed successfully!")
    print(f"Models saved in 'models/' directory")
    print(f"Best model: {trainer.best_model_name}")

if __name__ == "__main__":
    main()