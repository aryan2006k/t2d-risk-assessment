# src/recommendations.py
import pandas as pd
import numpy as np

class T2DRecommendationEngine:
    """Generate personalized lifestyle recommendations based on risk factors"""
    
    def __init__(self):
        self.recommendation_database = self.load_recommendation_database()
        
    def load_recommendation_database(self):
        """Load recommendation templates"""
        recommendations = {
            'BMI': {
                'high': {
                    'condition': lambda x: x > 30,
                    'recommendations': [
                        "Consider a weight management program targeting 5-10% body weight reduction",
                        "Consult with a registered dietitian for personalized meal planning",
                        "Aim for 150+ minutes of moderate-intensity exercise weekly",
                        "Track daily calorie intake using a food diary or mobile app"
                    ],
                    'priority': 'high'
                },
                'moderate': {
                    'condition': lambda x: 25 <= x <= 30,
                    'recommendations': [
                        "Maintain current weight or aim for modest weight reduction",
                        "Focus on increasing physical activity levels",
                        "Choose whole grains and lean proteins"
                    ],
                    'priority': 'medium'
                }
            },
            'Glucose': {
                'high': {
                    'condition': lambda x: x > 125,
                    'recommendations': [
                        "Schedule immediate consultation with healthcare provider",
                        "Monitor blood glucose levels regularly",
                        "Reduce intake of simple carbohydrates and sugary drinks",
                        "Consider continuous glucose monitoring"
                    ],
                    'priority': 'critical'
                },
                'prediabetic': {
                    'condition': lambda x: 100 <= x <= 125,
                    'recommendations': [
                        "Implement dietary modifications to control blood sugar",
                        "Increase fiber intake to 25-30g daily",
                        "Space meals evenly throughout the day",
                        "Consider joining a diabetes prevention program"
                    ],
                    'priority': 'high'
                }
            },
            'PhysicalActivity': {
                'low': {
                    'condition': lambda x: x < 2.5,
                    'recommendations': [
                        "Start with 10-minute walks after meals",
                        "Gradually increase to 150 minutes of moderate activity weekly",
                        "Try activities you enjoy: swimming, dancing, or cycling",
                        "Use a fitness tracker to monitor daily steps (target: 10,000)"
                    ],
                    'priority': 'high'
                }
            },
            'DietQuality': {
                'poor': {
                    'condition': lambda x: x < 5,
                    'recommendations': [
                        "Adopt a Mediterranean or DASH diet pattern",
                        "Increase vegetable intake to 5 servings daily",
                        "Replace processed foods with whole food alternatives",
                        "Meal prep on weekends to ensure healthy options"
                    ],
                    'priority': 'high'
                }
            },
            'StressLevel': {
                'high': {
                    'condition': lambda x: x > 7,
                    'recommendations': [
                        "Practice stress management techniques (meditation, yoga)",
                        "Ensure 7-9 hours of quality sleep nightly",
                        "Consider counseling or stress management programs",
                        "Schedule regular relaxation activities"
                    ],
                    'priority': 'medium'
                }
            },
            'SleepHours': {
                'insufficient': {
                    'condition': lambda x: x < 6,
                    'recommendations': [
                        "Establish consistent sleep schedule",
                        "Create a relaxing bedtime routine",
                        "Limit screen time 1 hour before bed",
                        "Evaluate for sleep disorders if problems persist"
                    ],
                    'priority': 'medium'
                }
            },
            'SmokingStatus': {
                'current': {
                    'condition': lambda x: x == 'Current',
                    'recommendations': [
                        "Enroll in smoking cessation program",
                        "Discuss nicotine replacement therapy with healthcare provider",
                        "Set a quit date and seek support from family/friends",
                        "Avoid triggers and find healthy alternatives to manage cravings"
                    ],
                    'priority': 'critical'
                }
            }
        }
        
        return recommendations
    
    def generate_recommendations(self, patient_data, risk_factors, risk_score):
        """Generate personalized recommendations based on patient data"""
        recommendations = []
        priorities = {'critical': [], 'high': [], 'medium': [], 'low': []}
        
        # Check each feature against recommendation criteria
        for feature, criteria_dict in self.recommendation_database.items():
            if feature in patient_data.columns:
                value = patient_data[feature].iloc[0] if isinstance(patient_data, pd.DataFrame) else patient_data[feature]
                
                for level, criteria in criteria_dict.items():
                    if criteria['condition'](value):
                        for rec in criteria['recommendations']:
                            priorities[criteria['priority']].append({
                                'category': feature,
                                'recommendation': rec,
                                'priority': criteria['priority']
                            })
        
        # Add general recommendations based on risk score
        if risk_score > 0.7:
            priorities['high'].append({
                'category': 'General',
                'recommendation': 'Schedule comprehensive health screening with your physician',
                'priority': 'high'
            })
            priorities['high'].append({
                'category': 'General',
                'recommendation': 'Consider enrollment in a diabetes prevention program',
                'priority': 'high'
            })
        
        # Compile final recommendations
        final_recommendations = {
            'critical': priorities['critical'],
            'high': priorities['high'][:5],  # Limit to top 5 high priority
            'medium': priorities['medium'][:3],  # Limit to top 3 medium priority
            'lifestyle_plan': self.create_lifestyle_plan(patient_data, risk_factors)
        }
        
        return final_recommendations
    
    def create_lifestyle_plan(self, patient_data, risk_factors):
        """Create a structured lifestyle modification plan"""
        plan = {
            'weekly_goals': [],
            'monthly_goals': [],
            'tracking_metrics': []
        }
        
        # Set goals based on risk factors
        if 'BMI' in risk_factors and patient_data['BMI'].iloc[0] > 25:
            plan['weekly_goals'].append("Exercise 150 minutes at moderate intensity")
            plan['monthly_goals'].append("Lose 2-4 pounds through diet and exercise")
            plan['tracking_metrics'].append("Daily weight measurement")
        
        if 'PhysicalActivity' in risk_factors:
            plan['weekly_goals'].append("Increase daily steps by 1000 each week")
            plan['tracking_metrics'].append("Daily step count")
        
        if 'DietQuality' in risk_factors:
            plan['weekly_goals'].append("Prepare 5 home-cooked meals")
            plan['monthly_goals'].append("Reduce processed food intake by 50%")
            plan['tracking_metrics'].append("Food diary")
        
        return plan
    
    def format_recommendations_report(self, recommendations, patient_id=None):
        """Format recommendations into a readable report"""
        report = f"# Personalized T2D Prevention Recommendations\n"
        if patient_id:
            report += f"**Patient ID**: {patient_id}\n\n"
        
        report += "## Priority Actions\n\n"
        
        # Critical recommendations
        if recommendations['critical']:
            report += "### 🔴 Critical (Immediate Action Required)\n"
            for rec in recommendations['critical']:
                report += f"- {rec['recommendation']}\n"
            report += "\n"
        
        # High priority
        if recommendations['high']:
            report += "### 🟠 High Priority\n"
            for rec in recommendations['high']:
                report += f"- {rec['recommendation']}\n"
            report += "\n"
        
        # Medium priority
        if recommendations['medium']:
            report += "### 🟡 Medium Priority\n"
            for rec in recommendations['medium']:
                report += f"- {rec['recommendation']}\n"
            report += "\n"
        
        # Lifestyle plan
        if recommendations['lifestyle_plan']:
            plan = recommendations['lifestyle_plan']
            report += "## Structured Lifestyle Plan\n\n"
            
            if plan['weekly_goals']:
                report += "### Weekly Goals\n"
                for goal in plan['weekly_goals']:
                    report += f"- {goal}\n"
                report += "\n"
            
            if plan['monthly_goals']:
                report += "### Monthly Goals\n"
                for goal in plan['monthly_goals']:
                    report += f"- {goal}\n"
                report += "\n"
            
            if plan['tracking_metrics']:
                report += "### Track These Metrics\n"
                for metric in plan['tracking_metrics']:
                    report += f"- {metric}\n"
        
        return report