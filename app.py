import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide"
)

# -----------------------------
# Load saved files
# -----------------------------
with open("loan_approval_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("onehot_encoder.pkl", "rb") as file:
    ohe = pickle.load(file)

with open("feature_columns.pkl", "rb") as file:
    feature_columns = pickle.load(file)


# -----------------------------
# Introduction
# -----------------------------
st.title("💰 Loan Approval Prediction")

st.write("""
### Welcome to the Loan Approval Prediction System

This application uses **Machine Learning** to predict whether a loan
application is likely to be approved.

Please enter the applicant information below.
""")

# -----------------------------
# Minimum Qualification
# -----------------------------
with st.expander("📋 View Minimum Qualification Requirements"):

    st.info("""
    ### Basic Eligibility Requirements

    The following are basic eligibility checks used in this
    educational project:

    - Age: 21 years or above
    - Credit Score: 600 or above
    - Applicant Income: ₹10,000 or above
    - Existing Loans: Maximum 5
    - DTI Ratio: 0.50 or below
    - Savings: Greater than ₹0

    ⚠️ These are demo/project criteria and are not actual banking rules.
    """)


# -----------------------------
# User Input
# -----------------------------
st.subheader("👤 Applicant Information")

col1, col2 = st.columns(2)

with col1:

    applicant_income = st.number_input(
        "Applicant Income (₹)",
        min_value=0.0,
        value=10000.0
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income (₹)",
        min_value=0.0,
        value=5000.0
    )

    employment_status = st.selectbox(
        "Employment Status",
        ["Salaried", "Self-employed", "Unemployed"]
    )

    age = st.number_input(
        "Age",
        min_value=21,
        max_value=100,
        value=35
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married"]
    )

    dependents = st.number_input(
        "Dependents",
        min_value=0,
        max_value=10,
        value=1
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=680
    )

    existing_loans = st.number_input(
        "Existing Loans",
        min_value=0,
        max_value=20,
        value=2
    )

    dti_ratio = st.number_input(
        "DTI Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.01
    )


with col2:

    savings = st.number_input(
        "Savings (₹)",
        min_value=0.0,
        value=10000.0
    )

    collateral_value = st.number_input(
        "Collateral Value (₹)",
        min_value=0.0,
        value=25000.0
    )

    loan_amount = st.number_input(
        "Loan Amount (₹)",
        min_value=0.0,
        value=20000.0
    )

    loan_term = st.selectbox(
        "Loan Term (months)",
        [12, 24, 36, 48, 60, 72, 84]
    )

    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Personal", "Car", "Business", "Education", "Home"]
    )

    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )

    education_level = st.selectbox(
        "Education Level",
        ["Graduate", "Not Graduate"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    employer_category = st.selectbox(
        "Employer Category",
        [
            "Private",
            "Government",
            "Business",
            "Self-employed",
            "Other"
        ]
    )


# -----------------------------
# Prediction Button
# -----------------------------
if st.button(
    "🔍 Check Eligibility & Predict Loan",
    use_container_width=True
):

    # -----------------------------
    # Minimum Qualification Check
    # -----------------------------

    eligibility_reasons = []

    if age < 21:
        eligibility_reasons.append(
            "Age must be at least 21 years."
        )

    if credit_score < 600:
        eligibility_reasons.append(
            "Credit Score must be at least 600."
        )

    if applicant_income < 10000:
        eligibility_reasons.append(
            "Applicant income must be at least ₹10,000."
        )

    if existing_loans > 5:
        eligibility_reasons.append(
            "Existing loans should not exceed 5."
        )

    if dti_ratio > 0.50:
        eligibility_reasons.append(
            "DTI Ratio should not exceed 50%."
        )

    if savings <= 0:
        eligibility_reasons.append(
            "Savings must be greater than ₹0."
        )

    # -----------------------------
    # Eligibility Failed
    # -----------------------------

    if eligibility_reasons:

        st.error(
            "❌ Applicant does not meet the minimum qualification."
        )

        st.subheader("⚠️ Requirements Not Met")

        for reason in eligibility_reasons:
            st.warning(reason)

    # -----------------------------
    # Eligibility Passed
    # -----------------------------

    else:

        st.success(
            "✅ Minimum qualification requirements satisfied!"
        )

        # -----------------------------
        # Create Input DataFrame
        # -----------------------------

        input_df = pd.DataFrame({

            "Applicant_Income": [applicant_income],

            "Coapplicant_Income": [coapplicant_income],

            "Employment_Status": [employment_status],

            "Age": [age],

            "Marital_Status": [marital_status],

            "Dependents": [dependents],

            "Credit_Score": [credit_score],

            "Existing_Loans": [existing_loans],

            "DTI_Ratio": [dti_ratio],

            "Savings": [savings],

            "Collateral_Value": [collateral_value],

            "Loan_Amount": [loan_amount],

            "Loan_Term": [loan_term],

            "Loan_Purpose": [loan_purpose],

            "Property_Area": [property_area],

            "Education_Level": [education_level],

            "Gender": [gender],

            "Employer_Category": [employer_category]
        })


        # -----------------------------
        # Education Encoding
        # -----------------------------

        input_df["Education_Level"] = input_df[
            "Education_Level"
        ].map({
            "Not Graduate": 0,
            "Graduate": 1
        })


        # -----------------------------
        # Categorical Columns
        # -----------------------------

        categorical_cols = [

            "Employment_Status",

            "Marital_Status",

            "Loan_Purpose",

            "Property_Area",

            "Gender",

            "Employer_Category"
        ]


        # -----------------------------
        # One Hot Encoding
        # -----------------------------

        encoded = ohe.transform(
            input_df[categorical_cols]
        )

        encoded_df = pd.DataFrame(
            encoded,
            columns=ohe.get_feature_names_out(
                categorical_cols
            )
        )


        # -----------------------------
        # Remove Categorical Columns
        # -----------------------------

        input_df = input_df.drop(
            columns=categorical_cols
        )


        # -----------------------------
        # Combine Data
        # -----------------------------

        input_df = pd.concat(
            [
                input_df.reset_index(drop=True),

                encoded_df.reset_index(drop=True)
            ],
            axis=1
        )


        # -----------------------------
        # Feature Engineering
        # -----------------------------

        input_df["DTI_Ratio_sq"] = (
            input_df["DTI_Ratio"] ** 2
        )

        input_df["Credit_Score_sq"] = (
            input_df["Credit_Score"] ** 2
        )


        # -----------------------------
        # Drop Original Columns
        # -----------------------------

        input_df = input_df.drop(
            columns=[
                "Credit_Score",
                "DTI_Ratio"
            ]
        )


        # -----------------------------
        # Match Training Columns
        # -----------------------------

        input_df = input_df.reindex(
            columns=feature_columns,
            fill_value=0
        )


        # -----------------------------
        # Scaling
        # -----------------------------

        input_scaled = scaler.transform(
            input_df
        )


        # -----------------------------
        # Machine Learning Prediction
        # -----------------------------

        prediction = model.predict(
            input_scaled
        )[0]


        # -----------------------------
        # Final Result
        # -----------------------------

        st.subheader("🤖 Machine Learning Result")

        if prediction == 1:

            st.success(
                "## 🎉 Loan Approved"
            )

            st.write(
                "The Machine Learning model predicts that "
                "the applicant is likely to be approved."
            )

            st.balloons()

        else:

            st.error(
                "## ❌ Loan Not Approved"
            )

            st.write(
                "The Machine Learning model predicts that "
                "the applicant is unlikely to be approved."
            )


# -----------------------------
# Footer
# -----------------------------

st.divider()

st.caption(
    "⚠️ Educational Machine Learning Project | "
    "Not a real banking or lending decision system."
)