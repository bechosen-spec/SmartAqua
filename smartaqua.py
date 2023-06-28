import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components

# Set the configuration option to disable the warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the trained models
with open('/home/boniface/Desktop/Projects/SmartAqua/decision_tree_model.pkl', 'rb') as f:
    decision_tree_model = pickle.load(f)

with open('/home/boniface/Desktop/Projects/SmartAqua/xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# Define the feature names
feature_names = ['temperature(C)', 'turbidity (NTU)', 'Dissolved Oxygen (g/ml)', 'PH', 'ammonia(g/ml)', 'nitrate(g/ml)']

# Set aquaculture theme
st.markdown(
    """
    <style>
    .aquaculture {
        background-color: #4CB3A2;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 24px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.title("Smart Aqua")
st.markdown('<p class="aquaculture">App for Fish Length and Weight Prediction</p>', unsafe_allow_html=True)

# User input
temperature = st.number_input("Temperature (C)")
turbidity = st.number_input("Turbidity (NTU)")
dissolved_oxygen = st.number_input("Dissolved Oxygen (g/ml)")
ph = st.number_input("pH")
ammonia = st.number_input("Ammonia (g/ml)")
nitrate = st.number_input("Nitrate (g/ml)")

# Predict button with catchy name
if st.button("Get Fish Predictions!"):
    # Create input data as a DataFrame
    input_data = pd.DataFrame([[temperature, turbidity, dissolved_oxygen, ph, ammonia, nitrate]], columns=feature_names)

    # Predictions
    fish_length_prediction = decision_tree_model.predict(input_data)
    fish_weight_prediction = xgb_model.predict(input_data)

    # Display predictions
    st.write("Fish Length Prediction:", fish_length_prediction[0])
    st.write("Fish Weight Prediction:", fish_weight_prediction[0])

    # Feature importances
    importances_length = pd.Series(decision_tree_model.feature_importances_, index=feature_names).sort_values()
    importances_weight = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values()

    # Plot feature importances for fish length
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances_length.values, y=importances_length.index, palette='viridis')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances for Fish Length')
    st.pyplot()

    # Plot feature importances for fish weight
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances_weight.values, y=importances_weight.index, palette='magma')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances for Fish Weight')
    st.pyplot()