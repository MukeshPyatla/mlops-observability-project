# 🚀 Streamlit Cloud Deployment Guide

This guide will help you deploy the MLOps Observability Dashboard to Streamlit Cloud.

## 📋 Prerequisites

1. **GitHub Account**: You need a GitHub account to host your code
2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **Repository**: This code should be in a GitHub repository

## 🎯 Quick Deployment

### Step 1: Prepare Your Repository

1. **Fork or Clone this Repository**
   ```bash
   git clone <repository-url>
   cd mlops-observability-project
   ```

2. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit for Streamlit Cloud deployment"
   git push origin main
   ```

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Create New App**
   - Click "New app"
   - Select your repository
   - Set the main file path to: `streamlit_app.py`
   - Click "Deploy"

3. **Wait for Deployment**
   - The first deployment may take 5-10 minutes
   - The app will automatically run the setup script
   - You'll see progress in the deployment logs

## 🔧 Configuration Options

### Environment Variables (Optional)

You can add these environment variables in Streamlit Cloud if needed:

```bash
# Optional: Set random seed for reproducible results
RANDOM_SEED=42

# Optional: Set model parameters
MODEL_RANDOM_STATE=42
```

### Advanced Configuration

In the Streamlit Cloud dashboard, you can configure:

- **Python version**: 3.9 or higher
- **Memory**: 1GB (default is sufficient)
- **Timeout**: 300 seconds (default)

## 📊 What Happens During Deployment

1. **Automatic Setup**: The app runs `setup.py` automatically
2. **Data Generation**: Creates synthetic customer data
3. **Model Training**: Trains the XGBoost churn prediction model
4. **Directory Creation**: Sets up necessary folders
5. **App Launch**: Starts the Streamlit dashboard

## 🐛 Troubleshooting

### Common Issues

1. **Setup Timeout**
   - The setup process may take up to 5 minutes
   - Check the deployment logs for progress
   - If it fails, try redeploying

2. **Memory Issues**
   - Increase memory allocation in Streamlit Cloud
   - The app uses about 500MB during setup

3. **Import Errors**
   - Ensure all dependencies are in `requirements.txt`
   - Check that package versions are compatible

### Debug Steps

1. **Check Deployment Logs**
   - Go to your app in Streamlit Cloud
   - Click "Manage app" → "Logs"
   - Look for error messages

2. **Test Locally First**
   ```bash
   pip install -r requirements.txt
   streamlit run streamlit_app.py
   ```

3. **Verify File Structure**
   ```
   mlops-observability-project/
   ├── streamlit_app.py
   ├── setup.py
   ├── requirements.txt
   └── .streamlit/config.toml
   ```

## 🎉 Post-Deployment

### Verify Your App

1. **Check the Dashboard**
   - Navigate through all pages
   - Test the churn predictor
   - Verify data analysis works

2. **Monitor Performance**
   - Check app response times
   - Monitor memory usage
   - Verify model predictions

### Share Your App

1. **Get the URL**
   - Your app will be available at: `https://your-app-name.streamlit.app`

2. **Share with Others**
   - Send the URL to your team
   - Add to your portfolio
   - Share on social media

## 🔄 Updates and Maintenance

### Updating Your App

1. **Make Changes Locally**
   ```bash
   # Edit your files
   git add .
   git commit -m "Update app"
   git push origin main
   ```

2. **Redeploy**
   - Streamlit Cloud automatically redeploys
   - Or manually trigger redeployment

### Monitoring

- **App Health**: Check Streamlit Cloud dashboard
- **Usage Analytics**: Available in Streamlit Cloud
- **Performance**: Monitor response times and errors

## 📞 Support

### Streamlit Cloud Support
- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Community Forum](https://discuss.streamlit.io/)

### This Project
- Check the main README.md for detailed documentation
- Create issues in the GitHub repository
- Review deployment logs for specific errors

## 🎯 Best Practices

1. **Keep Dependencies Updated**
   - Regularly update `requirements.txt`
   - Test locally before deploying

2. **Optimize Performance**
   - Use `@st.cache_data` for expensive operations
   - Minimize memory usage during setup

3. **Security**
   - Don't commit sensitive data
   - Use environment variables for secrets
   - Keep dependencies secure

4. **Documentation**
   - Update README.md with changes
   - Document any custom configurations
   - Provide clear usage instructions

---

**Happy Deploying! 🚀** 