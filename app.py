import streamlit as st
import pickle
import pandas as pd
import sklearn 



# Page Config

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💳",
    layout="centered"
)


# Custom CSS

st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            background: linear-gradient(90deg, #e6f7ff, #f4f9ff);
            padding: 16px;
            border-radius: 12px;
            color: #003366;
            font-size: 32px;
            font-weight: 800;
            margin-bottom: 0px;
        }
        .subheader-text {
            text-align: center;
            color: #555555;
            font-size: 15px;
            margin-bottom: 25px;
        }
        .card {
            background-color: #ffffff;
            padding: 18px 18px 10px 18px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            border: 1px solid #f0f2f6;
        }
        .result-box-approved {
            background-color: #e6ffed;
            color: #135200;
            padding: 18px;
            border-radius: 10px;
            text-align: center;
            font-size: 20px;
            font-weight: 700;
            border: 1px solid #b7eb8f;
            margin-top: 16px;
        }
        .result-box-denied {
            background-color: #fff1f0;
            color: #a8071a;
            padding: 18px;
            border-radius: 10px;
            text-align: center;
            font-size: 20px;
            font-weight: 700;
            border: 1px solid #ffa39e;
            margin-top: 16px;
        }
        .ratio-chip {
            display: inline-block;
            background-color: #f0f5ff;
            padding: 6px 12px;
            border-radius: 999px;
            font-size: 13px;
            color: #1d39c4;
            font-weight: 600;
            margin-top: 4px;
        }
        .section-title {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# App Title

st.markdown("<div class='main-title'>Loan Approval Predictor</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subheader-text'>Fill in the applicant details below and evaluate the likelihood of loan approval.</div>",
    unsafe_allow_html=True
)

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

# Input Layout

st.markdown("<div class='section-title'>Applicant Details</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Loan & Credit Profile**", unsafe_allow_html=True)

        loan_amount = st.number_input(
            "Requested Loan Amount",
            min_value=0,
            max_value=1_000_000,
            step=1000,
            value=10_000,
            help="Total amount of the loan being requested."
        )

        fico_score = st.slider(
            "FICO Score",
            min_value=300,
            max_value=850,
            value=700,
            help="Higher scores generally indicate lower credit risk."
        )

        ever_bankrupt_or_foreclose = st.selectbox(
            "Ever Bankrupt or Foreclosed?",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Income & Housing Profile**", unsafe_allow_html=True)

        monthly_income = st.number_input(
            "Monthly Gross Income",
            min_value=1.0,  # avoids division by zero
            max_value=1_000_000.0,
            step=100.0,
            value=5_000.0,
            help="Total monthly income before taxes."
        )

        monthly_housing_payment = st.number_input(
            "Monthly Housing Payment",
            min_value=0.0,
            max_value=1_000_000.0,
            step=50.0,
            value=1_500.0,
            help="Monthly rent or mortgage payments."
        )

        housing_to_income_ratio = monthly_housing_payment / monthly_income

        st.metric(
            label="Housing-to-Income Ratio",
            value=f"{housing_to_income_ratio:.2f}"
        )
        st.markdown(
            f"<span class='ratio-chip'>Ratio: {housing_to_income_ratio:.2f} (Housing ÷ Income)</span>",
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)


# Categorical Info 

st.markdown("")
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Loan Context & Employment**", unsafe_allow_html=True)

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

    st.markdown("</div>", unsafe_allow_html=True)


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

input_encoded = pd.get_dummies(
    input_raw,
    columns=["Reason", "Employment_Status", "Employment_Sector", "Lender"]
)

input_encoded = input_encoded.reindex(columns=SELECTED_FEATURES, fill_value=0)

# Predict Button & Result Display
st.markdown("---")
center_col = st.columns([1, 2, 1])[1]

with center_col:
    if st.button("💡 Evaluate Loan"):
        result = model.predict(input_encoded)[0]

        if result == 1:
            st.markdown(
                "<div class='result-box-approved'>🎉 You got approved!</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='result-box-denied'>❌ You did not get approved.</div>",
                unsafe_allow_html=True
            )
