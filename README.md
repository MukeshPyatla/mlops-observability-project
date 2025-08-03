# 📊 MLOps Observability Dashboard

A comprehensive MLOps observability platform built with Streamlit that demonstrates customer churn prediction, model monitoring, and data drift detection.

## 🚀 Live Demo

This project is deployed on Streamlit Cloud and ready to use! The dashboard provides:

- **🔮 Real-time Churn Predictions**: Interactive customer churn prediction with feature analysis
- **📈 Data Analysis**: Comprehensive data visualization and correlation analysis
- **🚨 Model Monitoring**: Real-time drift detection and performance monitoring
- **📊 Reports & Analytics**: Detailed reports on model performance and business impact

## 🏗️ Project Structure

```
mlops-observability-project/
├── streamlit_app.py          # Main Streamlit application
├── setup.py                  # Data generation and model training script
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── src/
│   ├── app.py               # Original FastAPI application
│   ├── data_generator.py    # Synthetic data generation
│   ├── train_model.py       # Model training pipeline
│   └── monitor.py           # Model monitoring utilities
└── README.md                # Project documentation
```

## 🛠️ Features

### 🔮 Churn Prediction
- Interactive form for customer data input
- Real-time churn probability calculation
- Visual risk assessment with gauge charts
- Feature importance analysis

### 📈 Data Analysis
- Comprehensive data visualization
- Feature correlation analysis
- Distribution plots for all features
- Statistical summaries

### 🚨 Model Monitoring
- Real-time drift detection
- Performance metrics tracking
- Alert system for anomalies
- Historical trend analysis

### 📊 Reports & Analytics
- Model performance reports
- Data drift analysis
- Feature importance reports
- Business impact assessment

## 🚀 Deployment on Streamlit Cloud

### Prerequisites
- A GitHub repository with this code
- A Streamlit Cloud account

### Deployment Steps

1. **Fork or Clone this Repository**
   ```bash
   git clone <repository-url>
   cd mlops-observability-project
   ```

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select this repository
   - Set the main file path to: `streamlit_app.py`
   - Click "Deploy"

3. **Automatic Setup**
   - The app will automatically generate synthetic data
   - Train the XGBoost churn prediction model
   - Set up monitoring infrastructure
   - No manual configuration required!

## 🏃‍♂️ Local Development

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mlops-observability-project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```

### Manual Setup (Optional)

If you want to run the setup manually:

```bash
python setup.py
```

This will:
- Generate synthetic customer data
- Train the XGBoost model
- Create necessary directories and files

## 📊 Model Details

### Features
- **TenureMonths**: Customer tenure in months
- **ContractType**: Contract type (Month-to-month, One year, Two year)
- **SupportTickets**: Number of support tickets
- **MonthlyCharge**: Monthly subscription charge

### Model Architecture
- **Algorithm**: XGBoost Classifier
- **Preprocessing**: One-hot encoding for categorical features
- **Pipeline**: Scikit-learn Pipeline with ColumnTransformer
- **Performance**: ~87% accuracy on synthetic data

## 🔧 Configuration

### Streamlit Configuration
The app uses a custom Streamlit configuration in `.streamlit/config.toml`:

```toml
[global]
developmentMode = false

[server]
headless = true
port = 8501

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
```

### Environment Variables
No environment variables are required for basic functionality. The app generates all necessary data automatically.

## 📈 Monitoring & Observability

### Data Drift Detection
- Compares current data distributions with reference data
- Uses Evidently AI for drift analysis
- Provides visual drift reports

### Model Performance Tracking
- Real-time accuracy monitoring
- Precision, recall, and F1-score tracking
- Performance trend visualization

### Alert System
- Automated alerts for drift detection
- Performance threshold monitoring
- Real-time notification system

## 🎯 Use Cases

### Customer Success Teams
- Identify high-risk customers
- Prioritize retention efforts
- Track intervention effectiveness

### Data Science Teams
- Monitor model performance
- Detect data drift
- Validate model assumptions

### Business Stakeholders
- Track business impact
- Revenue optimization
- Customer retention metrics

## 🔍 API Documentation

The original FastAPI application is available in `src/app.py` for API-based deployments.

### Endpoints
- `GET /`: Health check
- `POST /predict/`: Churn prediction endpoint

### Example Request
```json
{
  "TenureMonths": 12,
  "ContractType": "Month-to-month",
  "SupportTickets": 2,
  "MonthlyCharge": 85.0
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Streamlit**: For the amazing web app framework
- **Evidently AI**: For model monitoring capabilities
- **XGBoost**: For the powerful gradient boosting algorithm
- **Plotly**: For interactive visualizations

## 📞 Support

For questions or issues:
- Create an issue in the GitHub repository
- Check the Streamlit Cloud deployment logs
- Review the documentation above

---

**Made with ❤️ for MLOps Observability**
