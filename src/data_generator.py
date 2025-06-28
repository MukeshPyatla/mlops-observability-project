import pandas as pd
import numpy as np
from faker import Faker

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

if __name__ == '__main__':
    # Generate the two datasets
    reference_data = generate_customer_data(num_customers=5000, data_type='reference')
    current_data = generate_customer_data(num_customers=5000, data_type='current')
    
    # Save them
    reference_data.to_csv('data/reference_data.csv', index=False)
    current_data.to_csv('data/current_data.csv', index=False)
    
    print("Reference and current datasets generated successfully in the 'data/' folder.")
