import streamlit as st
import pandas as pd
import numpy as np
from src.models.model_loader import load_models
from src.preprocessing.data_preprocessor import preprocess_input
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Food Poisoning Prediction System",
    page_icon="🍽️",
    layout="wide"
)

# Title and description
st.title("🍽️ Food Poisoning Prediction System")
st.markdown("""
This application helps predict the risk of food poisoning based on various food characteristics 
and environmental conditions using advanced machine learning models.
""")

# Sidebar for model selection
st.sidebar.header("Model Selection")
model_type = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "XGBoost"]
)

# Main input form
st.header("Input Parameters")

col1, col2 = st.columns(2)

with col1:
    # Food characteristics
    food_type = st.selectbox(
        "Food Type",
        ["Meat", "Seafood", "Dairy", "Produce", "Processed Food"]
    )
    
    storage_temp = st.slider(
        "Storage Temperature (°C)",
        -20.0, 40.0, 4.0
    )
    
    storage_time = st.number_input(
        "Storage Time (hours)",
        min_value=0,
        max_value=168,
        value=24
    )

with col2:
    # Environmental conditions
    humidity = st.slider(
        "Relative Humidity (%)",
        0.0, 100.0, 50.0
    )
    
    ph_level = st.slider(
        "pH Level",
        0.0, 14.0, 7.0
    )
    
    cooking_temp = st.slider(
        "Cooking Temperature (°C)",
        0.0, 200.0, 100.0
    )

# Additional parameters
st.subheader("Additional Parameters")
col3, col4 = st.columns(2)

with col3:
    packaging_type = st.selectbox(
        "Packaging Type",
        ["Vacuum Sealed", "Plastic Wrap", "Cardboard", "None"]
    )
    
    preservatives = st.checkbox("Contains Preservatives")

with col4:
    handling_score = st.slider(
        "Food Handling Score (1-10)",
        1, 10, 5
    )
    
    cross_contamination = st.checkbox("Risk of Cross Contamination")

# Prediction button
if st.button("Predict Food Poisoning Risk"):
    # Create input dataframe
    input_data = pd.DataFrame({
        'food_type': [food_type],
        'storage_temp': [storage_temp],
        'storage_time': [storage_time],
        'humidity': [humidity],
        'ph_level': [ph_level],
        'cooking_temp': [cooking_temp],
        'packaging_type': [packaging_type],
        'preservatives': [preservatives],
        'handling_score': [handling_score],
        'cross_contamination': [cross_contamination]
    })
    
    # Preprocess input
    processed_input = preprocess_input(input_data)
    
    # Load and use selected model
    models = load_models()
    model = models[model_type]
    prediction = model.predict_proba(processed_input)[0][1]
    
    # Display results
    st.header("Prediction Results")
    
    # Risk gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Food Poisoning Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))
    
    st.plotly_chart(fig)
    
    # Risk level and recommendations
    if prediction < 0.3:
        st.success("Low Risk: Food is likely safe to consume.")
    elif prediction < 0.7:
        st.warning("Moderate Risk: Take extra precautions when handling this food.")
    else:
        st.error("High Risk: Avoid consuming this food or ensure proper handling.")
    
    # Feature importance visualization
    st.subheader("Feature Importance")
    importance_data = pd.DataFrame({
        'Feature': processed_input.columns,
        'Importance': model.feature_importances_
    })
    importance_data = importance_data.sort_values('Importance', ascending=False)
    
    fig2 = px.bar(importance_data, x='Feature', y='Importance',
                  title='Feature Importance in Prediction')
    st.plotly_chart(fig2)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Developed with ❤️ for food safety</p>
    <p>Data sources: FSIS, FDA, Restaurant Inspection Data</p>
</div>
""", unsafe_allow_html=True) 