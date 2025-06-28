# MLOps Observability Platform with Drift Detection

This project demonstrates a critical component of a mature MLOps lifecycle: **automated model monitoring**. It builds a pipeline that continuously observes a deployed machine learning model, detects when its performance is degrading due to "drift," and generates a detailed report for analysis.

## 1. The Business Problem

Machine learning models are not static. Once a model is deployed to production, its performance can silently degrade as the real-world data it receives starts to differ from the data it was trained on. This is called **model drift**, and it can lead to inaccurate predictions, poor business decisions, and significant financial risk. The challenge is to detect this drift automatically and proactively.

## 2. The Solution

This project implements an MLOps observability pipeline that:
1.  **Simulates a live environment** with a trained churn prediction model served via an API.
2.  **Generates "current" production data** that has different characteristics (data drift) and relationships (concept drift) than the original training data.
3.  **Runs an automated monitoring script** that compares the statistical properties of the reference (training) data against the current (live) data.
4.  **Uses the `Evidently AI` library** to perform sophisticated statistical tests to detect both data and concept drift.
5.  **Generates a comprehensive HTML report** with detailed visualizations and analysis of any detected drift, which could be used to trigger alerts or an automated retraining pipeline.

## 3. Tech Stack

* **Programming Language:** Python
* **Core Libraries:** Pandas, Scikit-learn, XGBoost
* **Model Monitoring:** Evidently AI
* **API Simulation:** FastAPI, Uvicorn

## 4. How to Run This Project

### Step 1: Clone the Repository & Install Dependencies

```bash
git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/mlops-observability-project.git
cd mlops-observability-project
pip install -r requirements.txt