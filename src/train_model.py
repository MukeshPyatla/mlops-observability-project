import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

def train_model():
    """Trains a churn prediction model on the reference dataset."""
    print("Loading reference data...")
    if not os.path.exists('data/reference_data.csv'):
        print("Reference data not found. Please run data_generator.py first.")
        return

    df = pd.read_csv('data/reference_data.csv')

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
    X_train.to_csv('data/reference_training_data.csv', index=False)

    print("Model trained and saved to 'models/churn_model.pkl'.")
    print("Reference training data saved for monitoring.")

if __name__ == '__main__':
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Run data generation first
    from data_generator import generate_customer_data
    reference_data = generate_customer_data(num_customers=5000, data_type='reference')
    reference_data.to_csv('data/reference_data.csv', index=False)
    print("Generated reference_data.csv")

    # Now train the model
    train_model()