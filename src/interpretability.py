# src/interpretability.py
import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class T2DInterpreter:
    """Interpretability module for T2D risk assessment"""
    
    def __init__(self, model, feature_names, training_data=None):
        self.model = model
        self.feature_names = feature_names
        self.training_data = training_data
        self.shap_explainer = None
        self.lime_explainer = None
        
    def initialize_shap(self, background_data=None):
        """Initialize SHAP explainer"""
        if background_data is None:
            background_data = self.training_data
        
        # Use different explainers based on model type
        model_name = type(self.model).__name__
        
        if 'Tree' in model_name or 'Forest' in model_name or 'XGB' in model_name or 'LGBM' in model_name:
            self.shap_explainer = shap.TreeExplainer(self.model)
        else:
            # Use KernelExplainer for other models
            if background_data is not None:
                # Sample background data for efficiency
                background_sample = shap.sample(background_data, min(100, len(background_data)))
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba, 
                    background_sample
                )
    
    def initialize_lime(self, training_data):
        """Initialize LIME explainer"""
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data.values if isinstance(training_data, pd.DataFrame) else training_data,
            feature_names=self.feature_names,
            class_names=['Low Risk', 'High Risk'],
            mode='classification'
        )
    
    def get_shap_values(self, X):
        """Calculate SHAP values for given samples"""
        if self.shap_explainer is None:
            self.initialize_shap()
        
        shap_values = self.shap_explainer.shap_values(X)
        
        # Handle different output formats
        if isinstance(shap_values, list):
            # Binary classification - take values for positive class
            shap_values = shap_values[1]
        
        return shap_values
    
    def get_feature_importance(self):
        """Get global feature importance from the model"""
        importance_dict = {}
        
        # Check if model has feature_importances_ attribute
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            importance_dict = dict(zip(self.feature_names, importances))
        elif hasattr(self.model, 'coef_'):
            # For linear models
            importances = np.abs(self.model.coef_[0])
            importance_dict = dict(zip(self.feature_names, importances))
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), 
                                    key=lambda x: x[1], reverse=True))
        
        return importance_dict
    
    def explain_prediction(self, X_sample, method='shap'):
        """Explain a single prediction"""
        if method == 'shap':
            return self.explain_prediction_shap(X_sample)
        elif method == 'lime':
            return self.explain_prediction_lime(X_sample)
        else:
            raise ValueError("Method must be 'shap' or 'lime'")
    
    def explain_prediction_shap(self, X_sample):
        """Explain prediction using SHAP"""
        if self.shap_explainer is None:
            self.initialize_shap()
        
        # Get SHAP values
        shap_values = self.shap_explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Get prediction
        prediction = self.model.predict_proba(X_sample)[0, 1]
        
        # Create explanation dictionary
        if len(shap_values.shape) == 1:
            feature_impacts = dict(zip(self.feature_names, shap_values))
        else:
            feature_impacts = dict(zip(self.feature_names, shap_values[0]))
        
        # Sort by absolute impact
        feature_impacts = dict(sorted(feature_impacts.items(), 
                                    key=lambda x: abs(x[1]), reverse=True))
        
        explanation = {
            'prediction': prediction,
            'feature_impacts': feature_impacts,
            'top_factors': list(feature_impacts.keys())[:5]
        }
        
        return explanation
    
    def explain_prediction_lime(self, X_sample):
        """Explain prediction using LIME"""
        if self.lime_explainer is None:
            if self.training_data is None:
                raise ValueError("Training data required for LIME initialization")
            self.initialize_lime(self.training_data)
        
        # Get LIME explanation
        if isinstance(X_sample, pd.DataFrame):
            X_sample = X_sample.values[0]
        elif len(X_sample.shape) == 2:
            X_sample = X_sample[0]
        
        exp = self.lime_explainer.explain_instance(
            X_sample,
            self.model.predict_proba,
            num_features=len(self.feature_names)
        )
        
        # Get prediction
        prediction = self.model.predict_proba(X_sample.reshape(1, -1))[0, 1]
        
        # Create explanation dictionary
        feature_impacts = dict(exp.as_list())
        
        explanation = {
            'prediction': prediction,
            'feature_impacts': feature_impacts,
            'top_factors': [f[0] for f in exp.as_list()[:5]]
        }
        
        return explanation
    
    def create_waterfall_plot(self, explanation):
        """Create waterfall plot for individual prediction explanation"""
        impacts = explanation['feature_impacts']
        
        # Get top 10 features
        top_features = list(impacts.items())[:10]
        
        # Create waterfall data
        features = [f[0] for f in top_features]
        values = [f[1] for f in top_features]
        
        # Create waterfall plot
        fig = go.Figure(go.Waterfall(
            name="Feature Impact",
            orientation="v",
            measure=["relative"] * len(features),
            x=features,
            text=[f"{v:.3f}" for v in values],
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title="Feature Impact on T2D Risk Prediction",
            xaxis_title="Features",
            yaxis_title="Impact on Prediction",
            showlegend=False,
            height=500
        )
        
        return fig
    
    def create_feature_importance_plot(self):
        """Create global feature importance plot"""
        importance = self.get_feature_importance()
        
        # Get top 15 features
        top_features = list(importance.items())[:15]
        
        fig = go.Figure([go.Bar(
            x=[f[1] for f in top_features],
            y=[f[0] for f in top_features],
            orientation='h',
            marker_color='lightblue'
        )])
        
        fig.update_layout(
            title="Global Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Features",
            height=500
        )
        
        return fig
    
    def generate_explanation_text(self, explanation, patient_data=None):
        """Generate human-readable explanation text"""
        prediction = explanation['prediction']
        impacts = explanation['feature_impacts']
        top_factors = explanation['top_factors'][:5]
        
        # Determine risk level
        if prediction < 0.3:
            risk_level = "Low"
        elif prediction < 0.6:
            risk_level = "Moderate"
        elif prediction < 0.8:
            risk_level = "High"
        else:
            risk_level = "Very High"
        
        # Build explanation text
        explanation_text = f"## Risk Assessment Results\n\n"
        explanation_text += f"**T2D Risk Score**: {prediction:.1%}\n"
        explanation_text += f"**Risk Level**: {risk_level}\n\n"
        
        explanation_text += "### Key Contributing Factors:\n\n"
        
        for i, factor in enumerate(top_factors, 1):
            impact = impacts[factor]
            if impact > 0:
                direction = "increases"
            else:
                direction = "decreases"
            
            explanation_text += f"{i}. **{factor}**: {direction} risk by {abs(impact):.3f}\n"
        
        return explanation_text


