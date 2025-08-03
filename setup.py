#!/usr/bin/env python3
"""
Setup script for MLOps Observability Project
This script generates data and trains the model for Streamlit Cloud deployment.
"""

import os
import sys
import pandas as pd
import numpy as np
from faker import Faker
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

def generate_customer_data(num_customers=2500, data_type='reference'):
    """
    Generates synthetic customer data for a subscription service.
    The 'current' data type will have different characteristics to simulate drift.
    """
    fake = Faker()
    data = []
    
    for i in range(num_customers):
        tenure_months = np.random.randint(1, 48)
        support_tickets = np.random.poisson(1)
        contract_type = 'Month-to-month'
        
        # Simulate data drift in the 'current' dataset
        if data_type == 'current':
            # In the new data, customers are newer and have more support issues
            tenure_months = np.random.randint(1, 24) 
            support_tickets = np.random.poisson(2)
            contract_type = np.random.choice(['Month-to-month', 'One year'], p=[0.9, 0.1])
        
        monthly_charge = 60 + (tenure_months * 0.5) + (support_tickets * 5) + np.random.normal(0, 5)
        
        churn_probability = 0.1
        if contract_type == 'Month-to-month': churn_probability += 0.3
        if support_tickets > 2: churn_probability += 0.25
        if tenure_months < 12: churn_probability += 0.15
            
        # Simulate concept drift in the 'current' dataset
        if data_type == 'current':
             # In the new reality, even short-term customers are less likely to churn
            if tenure_months < 12: churn_probability -= 0.1
            
        churn = 1 if np.random.rand() < churn_probability else 0

        data.append([
            tenure_months,
            contract_type,
            support_tickets,
            monthly_charge,
            churn
        ])

    df = pd.DataFrame(data, columns=[
        'TenureMonths', 'ContractType', 'SupportTickets', 'MonthlyCharge', 'Churn'
    ])
    
    return df

def train_model():
    """Trains a churn prediction model on the reference dataset."""
    print("Loading reference data...")
    
    # Generate reference data
    df = generate_customer_data(num_customers=5000, data_type='reference')

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create a preprocessor for categorical and numerical features
    categorical_features = ['ContractType']
    numerical_features = ['TenureMonths', 'SupportTickets', 'MonthlyCharge']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Create the model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])

    print("Training XGBoost model...")
    model_pipeline.fit(X_train, y_train)

    # Save the model and the reference data for monitoring
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_pipeline, 'models/churn_model.pkl')
    
    # Save the training set used for reference in drift detection
    os.makedirs('data', exist_ok=True)
    X_train.to_csv('data/reference_training_data.csv', index=False)

    print("Model trained and saved to 'models/churn_model.pkl'.")
    print("Reference training data saved for monitoring.")

def main():
    """Main setup function."""
    print("🚀 Setting up MLOps Observability Project...")
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Generate data
    print("📊 Generating synthetic customer data...")
    reference_data = generate_customer_data(num_customers=5000, data_type='reference')
    current_data = generate_customer_data(num_customers=5000, data_type='current')
    
    reference_data.to_csv('data/reference_data.csv', index=False)
    current_data.to_csv('data/current_data.csv', index=False)
    print("✅ Data generated successfully!")
    
    # Train model
    print("🤖 Training churn prediction model...")
    train_model()
    print("✅ Model training completed!")
    
    print("🎉 Setup completed successfully!")
    print("📁 Generated files:")
    print("   - data/reference_data.csv")
    print("   - data/current_data.csv")
    print("   - data/reference_training_data.csv")
    print("   - models/churn_model.pkl")

if __name__ == "__main__":
    main() 