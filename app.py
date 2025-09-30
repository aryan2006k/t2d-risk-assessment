# app.py
import gradio as gr
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from src.preprocessing import T2DPreprocessor
from src.interpretability import T2DInterpreter
from src.recommendations import T2DRecommendationEngine
import warnings
warnings.filterwarnings('ignore')

# Load models and preprocessor
print("Loading models...")
model = joblib.load('models/best_model.pkl')
preprocessor = T2DPreprocessor.load_preprocessor('models/preprocessor.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# Initialize interpreter and recommendation engine
interpreter = T2DInterpreter(model, feature_names)
recommender = T2DRecommendationEngine()

def predict_risk(age, gender, ethnicity, bmi, glucose, hba1c, 
                 systolic_bp, diastolic_bp, total_chol, hdl, ldl, 
                 triglycerides, physical_activity, diet_quality, 
                 smoking_status, alcohol_consumption, sleep_hours, 
                 stress_level, family_diabetes, family_hypertension,
                 has_hypertension, has_obesity, has_pcos, had_gestational_diabetes):
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Ethnicity': [ethnicity],
        'BMI': [bmi],
        'Glucose': [glucose],
        'HbA1c': [hba1c],
        'SystolicBP': [systolic_bp],
        'DiastolicBP': [diastolic_bp],
        'TotalCholesterol': [total_chol],
        'HDL': [hdl],
        'LDL': [ldl],
        'Triglycerides': [triglycerides],
        'PhysicalActivity': [physical_activity],
        'DietQuality': [diet_quality],
        'SmokingStatus': [smoking_status],
        'AlcoholConsumption': [alcohol_consumption],
        'SleepHours': [sleep_hours],
        'StressLevel': [stress_level],
        'FamilyHistoryDiabetes': [1 if family_diabetes else 0],
        'FamilyHistoryHypertension': [1 if family_hypertension else 0],
        'Hypertension': [1 if has_hypertension else 0],
        'Obesity': [1 if has_obesity else 0],
        'PCOS': [1 if has_pcos else 0],
        'GestationalDiabetes': [1 if had_gestational_diabetes else 0]
    })
    
    # Preprocess input
    processed_input = preprocessor.handle_missing_values(input_data.copy())
    processed_input = preprocessor.encode_categorical_features(processed_input)
    processed_input = preprocessor.create_feature_interactions(processed_input)
    
    # Align with training features
    for col in feature_names:
        if col not in processed_input.columns:
            processed_input[col] = 0
    processed_input = processed_input[feature_names]
    
    # Scale features
    if 'standard' in preprocessor.scalers:
        scaler = preprocessor.scalers['standard']
        numerical_cols = processed_input.select_dtypes(include=[np.number]).columns
        processed_input[numerical_cols] = scaler.transform(processed_input[numerical_cols])
    
    # Make prediction
    risk_probability = model.predict_proba(processed_input)[0, 1]
    
    # Get explanation
    interpreter.initialize_shap(processed_input)
    explanation = interpreter.explain_prediction(processed_input, method='shap')
    
    # Generate recommendations
    risk_factors = explanation['top_factors'][:5]
    recommendations = recommender.generate_recommendations(
        input_data, risk_factors, risk_probability
    )
    
    # Create visualizations
    # 1. Risk gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_probability * 100,
        title={'text': "T2D Risk Score (%)"},
        delta={'reference': 30},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkred" if risk_probability > 0.7 else "orange" if risk_probability > 0.3 else "green"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig_gauge.update_layout(height=300)
    
    # 2. Feature impact plot
    impacts = explanation['feature_impacts']
    top_impacts = dict(list(impacts.items())[:10])
    
    fig_impact = go.Figure([go.Bar(
        x=list(top_impacts.values()),
        y=list(top_impacts.keys()),
        orientation='h',
        marker_color=['red' if v > 0 else 'green' for v in top_impacts.values()]
    )])
    
    fig_impact.update_layout(
        title="Top Risk Factors",
        xaxis_title="Impact on Risk",
        yaxis_title="Features",
        height=400
    )
    
    # Format recommendations
    rec_text = recommender.format_recommendations_report(recommendations)
    
    # Risk interpretation
    if risk_probability < 0.3:
        risk_level = "Low Risk"
        risk_message = "Your current risk for developing T2D is low. Continue maintaining healthy lifestyle habits."
    elif risk_probability < 0.6:
        risk_level = "Moderate Risk"
        risk_message = "You have moderate risk for T2D. Consider implementing the recommended lifestyle modifications."
    elif risk_probability < 0.8:
        risk_level = "High Risk"
        risk_message = "You are at high risk for developing T2D. Please consult with your healthcare provider and follow the recommendations."
    else:
        risk_level = "Very High Risk"
        risk_message = "You have very high risk for T2D. Immediate medical consultation and intervention recommended."
    
    return risk_level, f"{risk_probability:.1%}", fig_gauge, fig_impact, rec_text

# Create Gradio interface
with gr.Blocks(title="T2D Risk Assessment") as demo:
    gr.Markdown("""
    # 🏥 Type 2 Diabetes Risk Assessment & Lifestyle Recommendations
    
    This AI-powered tool provides personalized T2D risk assessment using interpretable machine learning.
    Enter your health information below for a comprehensive risk analysis and tailored recommendations.
    
    **Note**: This tool is for educational purposes only and does not replace professional medical advice.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 👤 Demographics")
            age = gr.Slider(18, 85, value=45, label="Age")
            gender = gr.Radio(["Male", "Female"], value="Male", label="Gender")
            ethnicity = gr.Dropdown(
                ["Caucasian", "African American", "Hispanic", "Asian", "Other"],
                value="Caucasian", label="Ethnicity"
            )
            bmi = gr.Slider(16, 45, value=25, step=0.1, label="BMI")
            
        with gr.Column(scale=1):
            gr.Markdown("### 🔬 Clinical Measurements")
            glucose = gr.Slider(70, 200, value=95, label="Fasting Glucose (mg/dL)")
            hba1c = gr.Slider(4.0, 10.0, value=5.5, step=0.1, label="HbA1c (%)")
            systolic_bp = gr.Slider(90, 180, value=120, label="Systolic BP")
            diastolic_bp = gr.Slider(60, 110, value=80, label="Diastolic BP")
            
        with gr.Column(scale=1):
            gr.Markdown("### 🧪 Lipid Profile")
            total_chol = gr.Slider(120, 350, value=200, label="Total Cholesterol")
            hdl = gr.Slider(20, 100, value=50, label="HDL Cholesterol")
            ldl = gr.Slider(50, 200, value=120, label="LDL Cholesterol")
            triglycerides = gr.Slider(50, 500, value=150, label="Triglycerides")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🏃 Lifestyle Factors")
            physical_activity = gr.Slider(0, 20, value=3, step=0.5, 
                                         label="Physical Activity (hours/week)")
            diet_quality = gr.Slider(0, 10, value=6, label="Diet Quality Score")
            smoking_status = gr.Radio(["Never", "Former", "Current"], 
                                     value="Never", label="Smoking Status")
            alcohol_consumption = gr.Slider(0, 20, value=2, 
                                          label="Alcohol (drinks/week)")
            sleep_hours = gr.Slider(4, 12, value=7, step=0.5, label="Sleep Hours")
            stress_level = gr.Slider(0, 10, value=5, label="Stress Level")
            
        with gr.Column(scale=1):
            gr.Markdown("### 🏥 Medical History")
            family_diabetes = gr.Checkbox(label="Family History of Diabetes")
            family_hypertension = gr.Checkbox(label="Family History of Hypertension")
            has_hypertension = gr.Checkbox(label="Diagnosed Hypertension")
            has_obesity = gr.Checkbox(label="Diagnosed Obesity")
            has_pcos = gr.Checkbox(label="PCOS (if applicable)")
            had_gestational_diabetes = gr.Checkbox(label="Gestational Diabetes (if applicable)")
    
    predict_btn = gr.Button("🔍 Assess Risk & Get Recommendations", variant="primary")
    
    with gr.Row():
        risk_level_output = gr.Textbox(label="Risk Level")
        risk_score_output = gr.Textbox(label="Risk Score")
    
    with gr.Row():
        gauge_plot = gr.Plot(label="Risk Visualization")
        impact_plot = gr.Plot(label="Contributing Factors")
    
    recommendations_output = gr.Markdown(label="Personalized Recommendations")
    
    predict_btn.click(
        fn=predict_risk,
        inputs=[age, gender, ethnicity, bmi, glucose, hba1c, 
                systolic_bp, diastolic_bp, total_chol, hdl, ldl, 
                triglycerides, physical_activity, diet_quality, 
                smoking_status, alcohol_consumption, sleep_hours, 
                stress_level, family_diabetes, family_hypertension,
                has_hypertension, has_obesity, has_pcos, had_gestational_diabetes],
        outputs=[risk_level_output, risk_score_output, gauge_plot, 
                impact_plot, recommendations_output]
    )
    
    gr.Markdown("""
    ---
    ### ⚠️ Disclaimer
    This tool is for educational and research purposes only. It should not be used as a substitute 
    for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare 
    providers for medical decisions.
    
    ### 📊 About the Model
    - Uses ensemble machine learning with XGBoost/Random Forest
    - Provides SHAP-based explanations for transparency
    - Trained on synthetic EHR data patterns
    - Generates evidence-based lifestyle recommendations
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()