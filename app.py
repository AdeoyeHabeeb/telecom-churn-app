import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Load trained model
loaded_model = joblib.load('random_forest_churn_model.pkl')

# --- Features and Label Encoders ---
all_categorical = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
                   'SeniorCitizen','Partner','Dependents','Contract','InternetService','PaymentMethod']
numerical_features = ['tenure','MonthlyCharges','TotalCharges']  # Added TotalCharges
model_features_ordered = all_categorical + numerical_features

unique_categories = {
    'OnlineSecurity': ['No', 'Yes', 'No internet service'],
    'OnlineBackup': ['No', 'Yes', 'No internet service'],
    'DeviceProtection': ['No', 'Yes', 'No internet service'],
    'TechSupport': ['No', 'Yes', 'No internet service'],
    'SeniorCitizen': ['No', 'Yes'],
    'Partner': ['No', 'Yes'],
    'Dependents': ['No', 'Yes'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'PaymentMethod': ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check']
}

label_encoders = {}
for col, categories in unique_categories.items():
    le = LabelEncoder()
    le.fit(categories)
    label_encoders[col] = le

# --- Prediction Function ---
def predict_churn(new_data: pd.DataFrame):
    new_data_processed = new_data.copy()
    
    if 'SeniorCitizen' in new_data_processed.columns and new_data_processed['SeniorCitizen'].dtype in ['int64', 'int32']:
        new_data_processed['SeniorCitizen'] = new_data_processed['SeniorCitizen'].map({1: 'Yes', 0: 'No'})
    
    for col in label_encoders:
        if col in new_data_processed.columns:
            try:
                new_data_processed[col] = label_encoders[col].transform(new_data_processed[col])
            except ValueError:
                st.warning(f"Invalid category for {col}. Allowed values: {unique_categories[col]}")
                return None
    
    new_data_processed = new_data_processed[model_features_ordered]
    
    y_pred = loaded_model.predict(new_data_processed)
    y_proba = loaded_model.predict_proba(new_data_processed)[:, 1]
    
    return {
        'Predicted_Class': y_pred[0],
        'Churn_Probability': y_proba[0]
    }

# --- Streamlit UI ---
st.set_page_config(page_title="Telecom Customer Churn Predictor", layout="wide", page_icon="📈")

st.markdown("""
<div style="background-color:#4B8BBE;padding:10px;border-radius:10px">
<h1 style="color:white;text-align:center;">📈 Telecom Customer Churn Predictor</h1>
<p style="color:white;text-align:center;font-size:18px;">Predict whether a customer is likely to churn and visualize the risk.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### Enter Customer Details:")

# --- Numerical Inputs with sliders + number input ---
col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.slider("Tenure (months):", min_value=0, max_value=72, value=32, step=1)
    tenure = st.number_input("Or enter exact Tenure:", min_value=0, max_value=72, value=tenure, step=1)

with col2:
    monthly_charges = st.slider("Monthly Charges ($):", min_value=18.25, max_value=118.75, value=64.85, step=0.25)
    monthly_charges = st.number_input("Or enter exact Monthly Charges ($):", min_value=18.25, max_value=118.75, value=monthly_charges, step=0.25, format="%.2f")

with col3:
    total_charges = st.slider("Total Charges ($):", min_value=18.8, max_value=8684.8, value=2290.35, step=1.0)
    total_charges = st.number_input("Or enter exact Total Charges ($):", min_value=18.8, max_value=8684.8, value=total_charges, step=0.25, format="%.2f")

# --- Categorical Inputs ---
col1, col2 = st.columns(2)

with col1:
    contract = st.selectbox("Contract Type:", unique_categories['Contract'])
    internet_service = st.selectbox("Internet Service:", unique_categories['InternetService'])
    online_security = st.selectbox("Online Security:", unique_categories['OnlineSecurity'])
    online_backup = st.selectbox("Online Backup:", unique_categories['OnlineBackup'])
    payment_method = st.selectbox("Payment Method:", unique_categories['PaymentMethod'])

with col2:
    device_protection = st.selectbox("Device Protection:", unique_categories['DeviceProtection'])
    tech_support = st.selectbox("Tech Support:", unique_categories['TechSupport'])
    senior_citizen = st.selectbox("Senior Citizen:", ['No','Yes'])
    partner = st.selectbox("Partner:", ['No','Yes'])
    dependents = st.selectbox("Dependents:", ['No','Yes'])

# --- Prepare Input DataFrame ---
new_customer = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'Contract': [contract],
    'InternetService': [internet_service],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'PaymentMethod': [payment_method],
    'SeniorCitizen': [senior_citizen],
    'Partner': [partner],
    'Dependents': [dependents]
})

# --- Predict Button ---
if st.button("Predict Churn"):
    result = predict_churn(new_customer)
    if result:
        # Class display
        if result['Churn_Probability'] > 0.7:
            st.markdown("<h3 style='color:red'>⚠️ High Churn Risk</h3>", unsafe_allow_html=True)
        elif result['Churn_Probability'] > 0.4:
            st.markdown("<h3 style='color:orange'>⚠️ Medium Churn Risk</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color:green'>✅ Low Churn Risk</h3>", unsafe_allow_html=True)
        
        # Probability Gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = result['Churn_Probability']*100,
            title = {'text': "Churn Probability (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if result['Churn_Probability']>0.7 else "orange" if result['Churn_Probability']>0.4 else "green"},
                'steps': [
                    {'range': [0, 40], 'color': "green"},
                    {'range': [40, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "red"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

