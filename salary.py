# ======================
# 📌 1. Imports
# ======================
import streamlit as st
import pandas as pd
import joblib
import base64

# ======================
# 📌 Bonus: Set Background Image
# ======================
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Change "data.jpg" to your background image filename (jpeg or png)
set_bg("background.jpeg")

# ======================
# 📌 2. Load files
# ======================
data = pd.read_csv("adult_encoded.csv")
model = joblib.load("neural_network_model.pkl")
encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")
feature_order = joblib.load("feature_order.pkl")  # exact feature order from training!

# ======================
# 📌 Sidebar
# ======================
st.sidebar.title("🧑‍💼 Employee Salary Prediction")
st.sidebar.markdown(
    """
    This app predicts whether a person's salary is `<=50K` or `>50K` based on various demographic and work-related features.
    """
)

if st.sidebar.checkbox("Show Raw Data"):
    st.sidebar.dataframe(data.head())

# ======================
# 📌 3. App Title & Intro
# ======================
st.title("🧑‍💼 Employee Salary Prediction App")
st.markdown("Predict whether a person's salary is `<=50K` or `>50K` based on various features.")

# ======================
# 📌 4. Original text categories
# ======================
workclass_options = [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
    "Local-gov", "State-gov", "Without-pay", "Never-worked"
]

education_options = [
    "Bachelors", "HS-grad", "Masters",
    "Some-college", "Assoc-acdm", "Assoc-voc",
    "Doctorate", "Prof-school"
]

marital_status_options = [
    "Married-civ-spouse", "Divorced", "Never-married",
    "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
]

occupation_options = [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
    "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
    "Transport-moving", "Priv-house-serv", "Protective-serv",
    "Armed-Forces"
]

relationship_options = [
    "Wife", "Own-child", "Husband", "Not-in-family",
    "Other-relative", "Unmarried"
]

race_options = [
    "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo",
    "Other", "Black"
]

gender_options = ["Female", "Male"]

native_country_options = [
    "United-States", "Cambodia", "England", "Puerto-Rico",
    "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India",
    "Japan", "Greece", "South", "China", "Cuba", "Iran",
    "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
    "Vietnam", "Mexico", "Portugal", "Ireland", "France",
    "Dominican-Republic", "Laos", "Ecuador", "Taiwan",
    "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua",
    "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
    "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"
]

# ======================
# 📌 5. Responsive Input Layout
# ======================
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 17, 90, 30, help="Select your age")
    workclass = st.selectbox("Workclass", workclass_options)
    education = st.selectbox("Education", education_options)
    marital_status = st.selectbox("Marital Status", marital_status_options)
    relationship = st.selectbox("Relationship", relationship_options)

with col2:
    occupation = st.selectbox("Occupation", occupation_options)
    race = st.selectbox("Race", race_options)
    gender = st.selectbox("Gender", gender_options)
    native_country = st.selectbox("Native Country", native_country_options)
    hours_per_week = st.slider("Hours per week", 1, 100, 40)

# Advanced options inside an expander
with st.expander("Advanced Features (Optional)"):
    capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
    educational_num = st.number_input("Educational Number", min_value=1, max_value=20, value=10)
    fnlwgt = st.number_input("Fnlwgt", min_value=0, value=100000)

# ======================
# 📌 6. Format input into DataFrame
# ======================
input_data = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'education': [education],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'native-country': [native_country],
    'hours-per-week': [hours_per_week],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'educational-num': [educational_num],
    'fnlwgt': [fnlwgt]
})

# ======================
# 📌 7. Encode categoricals with label encoders
# ======================
for col in input_data.columns:
    if col in encoders:
        valid_classes = encoders[col].classes_
        input_data[col] = input_data[col].apply(
            lambda x: encoders[col].transform([x])[0] if x in valid_classes else -1
        )

# ======================
# 📌 8. Reorder features exactly as during training
# ======================
input_data = input_data[feature_order]

# ======================
# 📌 9. Scale input
# ======================
input_scaled = scaler.transform(input_data)

# ======================
# 📌 10. Prediction
# ======================
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    if prediction == 1:
        st.success(f"🎉 Predicted Salary: >50K")
    else:
        st.error(f"💼 Predicted Salary: <=50K")

    st.markdown(f"**Confidence:**")
    st.write(f"Probability of >50K: {proba[1]*100:.2f}%")
    st.write(f"Probability of <=50K: {proba[0]*100:.2f}%")
