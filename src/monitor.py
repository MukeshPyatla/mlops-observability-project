 pd
import joblib
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
import os

def monitor_drift():
    """
    Generates a drift report by comparing reference data with current data.
    """
    print("Starting drift monitoring process...")

    # Load the reference data (the data the model was trained on)
    try:
        reference_data = pd.read_csv('data/reference_training_data.csv')
    except FileNotFoundError:
        print("Error: Reference data not found. Please run train_model.py first.")
        return

    # Generate new "live" data to simulate the current production data
    print("Generating new 'current' data to simulate production traffic...")
    from data_generator import generate_customer_data
    current_data = generate_customer_data(num_customers=5000, data_type='current')

    # Load the trained model to get predictions
    try:
        model = joblib.load('models/churn_model.pkl')
    except FileNotFoundError:
        print("Error: Model not found. Please run train_model.py first.")
        return
        
    # Add predictions to our dataframes for classification performance analysis
    reference_data['prediction'] = model.predict(reference_data)
    current_data['prediction'] = model.predict(current_data)

    # Rename the actual churn column to 'target' for Evidently
    reference_data.rename(columns={'Churn': 'target'}, inplace=True)
    current_data.rename(columns={'Churn': 'target'}, inplace=True)

    print("Generating Evidently AI drift report...")
    # Create a report object
    drift_report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPreset(),
    ])

    # Run the report
    drift_report.run(reference_data=reference_data, current_data=current_data)

    # Save the report
    os.makedirs('reports', exist_ok=True)
    report_path = 'reports/model_drift_report.html'
    drift_report.save_html(report_path)

    print(f"Drift report saved successfully to '{report_path}'.")
    print("Open the HTML file in your browser to view the results.")

if __name__ == '__main__':
    monitor_drift()
