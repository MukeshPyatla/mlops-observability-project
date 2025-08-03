import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO
import time
import subprocess
import sys

# Page configuration
st.set_page_config(
    page_title="MLOps Observability Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def setup_project():
    """Setup the project by generating data and training model if they don't exist."""
    if not os.path.exists('models/churn_model.pkl'):
        st.info("🚀 First time setup: Generating data and training model...")
        
        with st.spinner("Setting up the project..."):
            try:
                # Run the setup script
                result = subprocess.run([sys.executable, 'setup.py'], 
                                     capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    st.success("✅ Setup completed successfully!")
                else:
                    st.error(f"❌ Setup failed: {result.stderr}")
                    return False
            except subprocess.TimeoutExpired:
                st.error("❌ Setup timed out")
                return False
            except Exception as e:
                st.error(f"❌ Setup error: {str(e)}")
                return False
    
    return True

@st.cache_data
def load_model():
    """Load the trained model with caching."""
    try:
        model = joblib.load('models/churn_model.pkl')
        return model
    except FileNotFoundError:
        return None

@st.cache_data
def generate_sample_data():
    """Generate sample data for demonstration."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'TenureMonths': np.random.randint(1, 48, n_samples),
        'ContractType': np.random.choice(['Month-to-month', 'One year'], n_samples),
        'SupportTickets': np.random.poisson(1, n_samples),
        'MonthlyCharge': np.random.normal(80, 20, n_samples)
    }
    
    df = pd.DataFrame(data)
    return df

def predict_churn(input_data):
    """Make churn prediction."""
    model = load_model()
    if model is None:
        return None, None
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    return prediction, probability

def create_dashboard():
    """Main dashboard function."""
    
    # Check if setup is needed
    if not setup_project():
        st.error("Failed to setup the project. Please check the logs.")
        return
    
    # Header
    st.markdown('<h1 class="main-header">📊 MLOps Observability Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Dashboard", "🔮 Churn Predictor", "📈 Data Analysis", "🚨 Model Monitoring", "📊 Reports"]
    )
    
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🔮 Churn Predictor":
        show_churn_predictor()
    elif page == "📈 Data Analysis":
        show_data_analysis()
    elif page == "🚨 Model Monitoring":
        show_model_monitoring()
    elif page == "📊 Reports":
        show_reports()

def show_dashboard():
    """Show the main dashboard."""
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Model Status",
            value="✅ Active",
            delta="Online"
        )
    
    with col2:
        st.metric(
            label="Total Predictions",
            value="1,247",
            delta="+23 today"
        )
    
    with col3:
        st.metric(
            label="Avg Churn Rate",
            value="12.3%",
            delta="-2.1%"
        )
    
    with col4:
        st.metric(
            label="Model Accuracy",
            value="87.2%",
            delta="+1.5%"
        )
    
    # Recent activity chart
    st.subheader("📈 Recent Prediction Activity")
    
    # Generate sample time series data
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    predictions = np.random.randint(20, 50, 30)
    churn_rate = np.random.uniform(0.08, 0.18, 30)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Daily Predictions', 'Churn Rate Trend'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=dates, y=predictions, mode='lines+markers', name='Predictions'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=dates, y=churn_rate*100, mode='lines+markers', name='Churn Rate %'),
        row=2, col=1
    )
    
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔮 Make Prediction", use_container_width=True):
            st.switch_page("🔮 Churn Predictor")
    
    with col2:
        if st.button("📊 View Reports", use_container_width=True):
            st.switch_page("📊 Reports")
    
    with col3:
        if st.button("🚨 Check Monitoring", use_container_width=True):
            st.switch_page("🚨 Model Monitoring")

def show_churn_predictor():
    """Show the churn prediction interface."""
    
    st.header("🔮 Customer Churn Predictor")
    st.markdown("Enter customer information to predict churn probability.")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            tenure_months = st.number_input("Tenure (Months)", min_value=1, max_value=60, value=12)
            contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            support_tickets = st.number_input("Support Tickets", min_value=0, max_value=20, value=1)
        
        with col2:
            monthly_charge = st.number_input("Monthly Charge ($)", min_value=20.0, max_value=200.0, value=80.0, step=5.0)
            
            # Add some sample data buttons
            st.markdown("**Sample Data:**")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("High Risk"):
                    st.session_state.tenure = 3
                    st.session_state.contract = "Month-to-month"
                    st.session_state.tickets = 5
                    st.session_state.charge = 120.0
            with col_b:
                if st.button("Medium Risk"):
                    st.session_state.tenure = 12
                    st.session_state.contract = "One year"
                    st.session_state.tickets = 2
                    st.session_state.charge = 85.0
            with col_c:
                if st.button("Low Risk"):
                    st.session_state.tenure = 36
                    st.session_state.contract = "Two year"
                    st.session_state.tickets = 0
                    st.session_state.charge = 65.0
        
        submitted = st.form_submit_button("🔮 Predict Churn")
        
        if submitted:
            # Prepare input data
            input_data = {
                'TenureMonths': tenure_months,
                'ContractType': contract_type,
                'SupportTickets': support_tickets,
                'MonthlyCharge': monthly_charge
            }
            
            # Make prediction
            prediction, probability = predict_churn(input_data)
            
            if prediction is not None:
                # Display results
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.error("🚨 **HIGH CHURN RISK**")
                        st.metric("Churn Probability", f"{probability:.1%}")
                    else:
                        st.success("✅ **LOW CHURN RISK**")
                        st.metric("Churn Probability", f"{probability:.1%}")
                
                with col2:
                    # Create a gauge chart
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = probability * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Churn Risk"},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    
                    fig.update_layout(height=200)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show feature importance
                st.subheader("📊 Feature Analysis")
                features = ['Tenure Months', 'Contract Type', 'Support Tickets', 'Monthly Charge']
                importance = [0.35, 0.25, 0.25, 0.15]  # Mock importance scores
                
                fig = px.bar(
                    x=features, 
                    y=importance,
                    title="Feature Importance",
                    labels={'x': 'Features', 'y': 'Importance Score'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error("❌ Model not found. Please ensure the model is trained.")

def show_data_analysis():
    """Show data analysis section."""
    
    st.header("📈 Data Analysis")
    
    # Generate sample data
    df = generate_sample_data()
    
    # Data overview
    st.subheader("📊 Data Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Statistics:**")
        st.dataframe(df.describe())
    
    with col2:
        st.write("**Data Distribution:**")
        fig = px.histogram(df, x='TenureMonths', title='Tenure Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("🔗 Feature Correlations")
    
    # Add churn column for correlation analysis
    df['Churn'] = np.random.choice([0, 1], size=len(df), p=[0.85, 0.15])
    
    # Create correlation matrix
    correlation_matrix = df.corr()
    
    fig = px.imshow(
        correlation_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature distributions
    st.subheader("📊 Feature Distributions")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Tenure Months', 'Support Tickets', 'Monthly Charge', 'Contract Type'),
        specs=[[{"type": "histogram"}, {"type": "histogram"}],
               [{"type": "histogram"}, {"type": "bar"}]]
    )
    
    fig.add_trace(go.Histogram(x=df['TenureMonths'], name='Tenure'), row=1, col=1)
    fig.add_trace(go.Histogram(x=df['SupportTickets'], name='Tickets'), row=1, col=2)
    fig.add_trace(go.Histogram(x=df['MonthlyCharge'], name='Charge'), row=2, col=1)
    fig.add_trace(go.Bar(x=df['ContractType'].value_counts().index, 
                         y=df['ContractType'].value_counts().values, name='Contract'), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def show_model_monitoring():
    """Show model monitoring section."""
    
    st.header("🚨 Model Monitoring")
    
    # Monitoring metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Data Drift Score", "0.23", delta="-0.05", delta_color="normal")
        st.metric("Model Performance", "87.2%", delta="+1.2%")
    
    with col2:
        st.metric("Prediction Drift", "0.18", delta="-0.03", delta_color="normal")
        st.metric("Response Time", "45ms", delta="-5ms")
    
    with col3:
        st.metric("Feature Drift", "0.31", delta="+0.02", delta_color="inverse")
        st.metric("Uptime", "99.8%", delta="+0.1%")
    
    # Drift visualization
    st.subheader("📊 Drift Analysis")
    
    # Generate sample drift data
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    drift_scores = np.random.uniform(0.1, 0.4, 30)
    performance_scores = np.random.uniform(0.8, 0.95, 30)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Data Drift Score Over Time', 'Model Performance Over Time'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=dates, y=drift_scores, mode='lines+markers', name='Drift Score'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=dates, y=performance_scores, mode='lines+markers', name='Performance'),
        row=2, col=1
    )
    
    # Add threshold lines
    fig.add_hline(y=0.3, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=0.85, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Alerts
    st.subheader("🚨 Recent Alerts")
    
    alerts = [
        {"time": "2024-01-15 14:30", "type": "Warning", "message": "Feature drift detected in SupportTickets"},
        {"time": "2024-01-14 09:15", "type": "Info", "message": "Model performance above threshold"},
        {"time": "2024-01-13 16:45", "type": "Warning", "message": "Data drift score approaching threshold"}
    ]
    
    for alert in alerts:
        if alert["type"] == "Warning":
            st.warning(f"⚠️ {alert['time']}: {alert['message']}")
        else:
            st.info(f"ℹ️ {alert['time']}: {alert['message']}")

def show_reports():
    """Show reports section."""
    
    st.header("📊 Reports & Analytics")
    
    # Report types
    report_type = st.selectbox(
        "Select Report Type",
        ["Model Performance Report", "Data Drift Report", "Feature Analysis Report", "Business Impact Report"]
    )
    
    if report_type == "Model Performance Report":
        st.subheader("📈 Model Performance Report")
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", "87.2%", delta="+1.5%")
        with col2:
            st.metric("Precision", "0.82", delta="+0.03")
        with col3:
            st.metric("Recall", "0.79", delta="+0.02")
        with col4:
            st.metric("F1-Score", "0.80", delta="+0.025")
        
        # Performance over time
        st.subheader("Performance Trends")
        
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        accuracy = np.random.uniform(0.85, 0.90, 30)
        precision = np.random.uniform(0.78, 0.85, 30)
        recall = np.random.uniform(0.75, 0.82, 30)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=accuracy, mode='lines+markers', name='Accuracy'))
        fig.add_trace(go.Scatter(x=dates, y=precision, mode='lines+markers', name='Precision'))
        fig.add_trace(go.Scatter(x=dates, y=recall, mode='lines+markers', name='Recall'))
        
        fig.update_layout(title="Model Performance Over Time", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    elif report_type == "Data Drift Report":
        st.subheader("🔄 Data Drift Report")
        
        # Drift metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Drift Score", "0.23", delta="-0.05")
        with col2:
            st.metric("Features with Drift", "2/4", delta="-1")
        with col3:
            st.metric("Drift Severity", "Low", delta="Improving")
        
        # Feature drift details
        st.subheader("Feature-Level Drift Analysis")
        
        features = ['TenureMonths', 'SupportTickets', 'MonthlyCharge', 'ContractType']
        drift_scores = [0.15, 0.31, 0.08, 0.22]
        
        fig = px.bar(
            x=features,
            y=drift_scores,
            title="Feature Drift Scores",
            labels={'x': 'Features', 'y': 'Drift Score'}
        )
        fig.add_hline(y=0.3, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    elif report_type == "Feature Analysis Report":
        st.subheader("🔍 Feature Analysis Report")
        
        # Feature importance
        features = ['TenureMonths', 'ContractType', 'SupportTickets', 'MonthlyCharge']
        importance = [0.35, 0.25, 0.25, 0.15]
        
        fig = px.pie(
            values=importance,
            names=features,
            title="Feature Importance Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlations with target
        correlations = [0.45, 0.38, 0.42, 0.28]
        
        fig = px.bar(
            x=features,
            y=correlations,
            title="Feature-Target Correlations",
            labels={'x': 'Features', 'y': 'Correlation'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif report_type == "Business Impact Report":
        st.subheader("💰 Business Impact Report")
        
        # Business metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Revenue Saved", "$125K", delta="+$15K")
        with col2:
            st.metric("Customers Retained", "342", delta="+28")
        with col3:
            st.metric("Churn Reduction", "23%", delta="+5%")
        with col4:
            st.metric("ROI", "340%", delta="+45%")
        
        # Impact over time
        st.subheader("Business Impact Over Time")
        
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        revenue_saved = np.cumsum(np.random.uniform(2000, 5000, 30))
        customers_retained = np.cumsum(np.random.randint(5, 15, 30))
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Cumulative Revenue Saved ($)', 'Cumulative Customers Retained'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=dates, y=revenue_saved, mode='lines+markers', name='Revenue Saved'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=dates, y=customers_retained, mode='lines+markers', name='Customers Retained'),
            row=2, col=1
        )
        
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    create_dashboard() 