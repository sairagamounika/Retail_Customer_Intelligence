
# Retail Customer Intelligence: Churn Prediction & Retention

## Problem Statement
In the competitive retail landscape, retaining existing customers is significantly more cost-effective than acquiring new ones. The goal of this project is to identify customers at risk of churning and target them with personalized retention strategies based on their value (CLV).

## Approach
1. **Data Prep & EDA**: Cleaned transactional data, engineered RFM features, and defined churn (90-day inactivity).
2. **Segmentation**: Grouped customers into 4 behavioral segments (e.g., "High-Value Loyal", "At-Risk") using K-Means clustering.
3. **CLV Modeling**: Predicted future value using BG/NBD and Gamma-Gamma probabilistic models.
4. **Churn Prediction**: Built a calibrated Random Forest classifier to predict churn probability.
5. **Retention Policy**: Combined Churn Risk + CLV to prioritize customers and recommend specific actions.

## Artifacts
- **Notebooks**:
  - `notebooks/01_data_prep_and_eda.ipynb`: Data cleaning and feature engineering.
  - `notebooks/02_customer_segmentation.ipynb`: Clustering and profiling.
  - `notebooks/03_CLV_modeling.ipynb`: Lifetime value estimation.
  - `notebooks/04_churn_modeling.ipynb`: Churn prediction, SHAP analysis, and policy definition.
- **App**:
  - `app/main.py`: FastAPI endpoint for real-time churn scoring.
  - `app/dashboard.py`: Streamlit dashboard for visualizing retention priorities.
- **Models**: Saved in `models/` (Random Forest, feature list).
- **Data**: Processed data in `data/processed/`.

## Results
- **Model Performance**:
  - ROC-AUC: ~0.72 (Random Forest)
  - Lift: Significant lift in top deciles, capturing a high percentage of churners.
- **Key Drivers**: Recency and Frequency are the strongest predictors of churn.
- **Business Impact**: Targeted retention strategy focuses budget on high-value, high-risk customers, ignoring low-value churners.

## How to Run

### Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Run the Notebooks
Execute notebooks 01-04 in order to generate data and train models.

### Run the API
Start the FastAPI server:
```bash
python app/main.py
```
Test via Swagger UI at `http://localhost:8000/docs`.

### Run the Dashboard
Launch the Streamlit app:
```bash
streamlit run app/dashboard.py
```
