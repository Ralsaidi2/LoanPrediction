import streamlit as st
import pickle
import pandas as pd
import sklearn 


# Load trained model

with open("loan_approval_model.pkl", "rb") as file:
    model = pickle.load(file)


SELECTED_FEATURES = [
    'Granted_Loan_Amount',
    'FICO_score',
    'Monthly_Gross_Income',
    'Monthly_Housing_Payment',
    'Ever_Bankrupt_or_Foreclose',
    'housing_to_income_ratio',
    'Reason_credit_card_refinancing',
    'Reason_home_improvement',
    'Reason_major_purchase',
    'Employment_Status_part_time',
    'Employment_Sector_consumer_discretionary',
    'Employment_Sector_consumer_staples',
    'Employment_Sector_energy',
    'Employment_Sector_financials',
    'Employment_Sector_health_care',
    'Employment_Sector_industrials',
    'Employment_Sector_information_technology',
    'Employment_Sector_materials',
    'Employment_Sector_utilities',
    'Lender_B',
    'Lender_C'
]


reason_map = {
    "Credit card refinancing": "credit_card_refinancing",
    "Home improvement": "home_improvement",
    "Major purchase": "major_purchase",
    "Debt consolidation": "debt_conslidation",  
    "Other": "other"                            
}

employment_status_map = {
    "Full-time": "full_time",   
    "Part-time": "part_time"
}

employment_sector_map = {
    "Consumer discretionary": "consumer_discretionary",
    "Consumer staples": "consumer_staples",
    "Energy": "energy",
    "Financials": "financials",
    "Health care": "health_care",
    "Industrials": "industrials",
    "Information technology": "information_technology",
    "Materials": "materials",
    "Utilities": "utilities",
    "Real estate / Other / None": "real_estate"  
}

lender_map = {
    "Lender A": "A",  
    "Lender B": "B",
    "Lender C": "C"
}

# ==============================
# App Title
# ==============================
st.markdown(
    """
    <h1 style='text-align: center; background-color: #e6f7ff; padding: 10px; color: #003366;'>
        <b>Loan Approval Predictor</b>
    </h1>
    """,
    unsafe_allow_html=True
)

st.header("Enter Loan Applicant's Details")

loan_amount = st.number_input(
    "Requested Loan Amount",
    min_value=0,
    max_value=1_000_000,
    step=1000,
    value=10_000
)

fico_score = st.slider(
    "FICO Score",
    min_value=300,
    max_value=850,
    value=700
)

monthly_income = st.number_input(
    "Monthly Gross Income",
    min_value=1.0,              # avoids division by zero, so no if needed
    max_value=1_000_000.0,
    step=100.0,
    value=5_000.0
)

monthly_housing_payment = st.number_input(
    "Monthly Housing Payment",
    min_value=0.0,
    max_value=1_000_000.0,
    step=50.0,
    value=1_500.0
)

ever_bankrupt_or_foreclose = st.selectbox(
    "Ever Bankrupt or Foreclosed?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

housing_to_income_ratio = monthly_housing_payment / monthly_income

reason_pretty = st.selectbox(
    "Reason for Loan",
    list(reason_map.keys())
)

employment_status_pretty = st.selectbox(
    "Employment Status",
    list(employment_status_map.keys())
)

employment_sector_pretty = st.selectbox(
    "Employment Sector",
    list(employment_sector_map.keys())
)

lender_pretty = st.selectbox(
    "Lender",
    list(lender_map.keys())
)


# Build input DataFrame 
input_raw = pd.DataFrame({
    "Granted_Loan_Amount": [loan_amount],
    "FICO_score": [fico_score],
    "Monthly_Gross_Income": [monthly_income],
    "Monthly_Housing_Payment": [monthly_housing_payment],
    "Ever_Bankrupt_or_Foreclose": [ever_bankrupt_or_foreclose],
    "housing_to_income_ratio": [housing_to_income_ratio],
    "Reason": [reason_map[reason_pretty]],
    "Employment_Status": [employment_status_map[employment_status_pretty]],
    "Employment_Sector": [employment_sector_map[employment_sector_pretty]],
    "Lender": [lender_map[lender_pretty]]
})

# One-hot encode using the coded columns
input_encoded = pd.get_dummies(
    input_raw,
    columns=["Reason", "Employment_Status", "Employment_Sector", "Lender"]
)

# Align to exactly the 21 selected model features
input_encoded = input_encoded.reindex(columns=SELECTED_FEATURES, fill_value=0)


# Predict Button

if st.button("Evaluate Loan"):
    result = model.predict(input_encoded)[0]

    messages = {
        1: " You got approved!",
        0: " You did not get approved."
    }

    st.markdown(
        f"<h2 style='text-align:center;'>{messages[result]}</h3>",
        unsafe_allow_html=True
    )
