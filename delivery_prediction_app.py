
import streamlit as st
import pandas as pd
import joblib

# --- Load Model and Feature Columns ---
# Make sure these files are in the same directory as your Streamlit app
log_reg_model = joblib.load('logistic_regression_model.pkl')
model_features = joblib.load('model_features.pkl')

# --- Function to preprocess new data and make prediction ---
def predict_delivery_on_time_app(model, new_data_point, feature_columns):
    new_df = pd.DataFrame([new_data_point])

    # Re-identify categorical columns from the original structure to ensure consistent encoding
    # For a deployed app, it's often better to hardcode these or load from a config if they are static.
    # Here, we infer them from the initial structure based on common values.
    categorical_cols_app = ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']

    # Apply one-hot encoding, ensuring all possible categories are considered
    # and aligning with the training features
    new_df_encoded = pd.get_dummies(new_df, columns=categorical_cols_app, drop_first=True)

    # Align columns of new_df_encoded with the feature_columns used during training
    for col in feature_columns:
        if col not in new_df_encoded.columns:
            new_df_encoded[col] = 0  # Add missing columns with 0
    new_df_encoded = new_df_encoded[feature_columns] # Ensure correct order and only relevant features

    prediction = model.predict(new_df_encoded)
    return prediction[0]

# --- Streamlit UI ---
st.title('Delivery On-Time Prediction Tool')
st.write('Enter the details below to predict if a delivery will be on time (1) or late (0).')

# Input fields for features
warehouse_block = st.selectbox('Warehouse Block', ['A', 'B', 'C', 'D', 'F'])
mode_of_shipment = st.selectbox('Mode of Shipment', ['Flight', 'Ship', 'Road'])
customer_care_calls = st.slider('Customer Care Calls', 2, 7, 4)
customer_rating = st.slider('Customer Rating', 1, 5, 3)
cost_of_product = st.number_input('Cost of the Product', min_value=96, max_value=310, value=200)
prior_purchases = st.slider('Prior Purchases', 2, 10, 3)
product_importance = st.selectbox('Product Importance', ['low', 'medium', 'high'])
gender = st.selectbox('Gender', ['F', 'M'])
discount_offered = st.number_input('Discount Offered', min_value=1, max_value=65, value=10)
weight_in_gms = st.number_input('Weight in Gms', min_value=1001, max_value=7846, value=3000)

# When the 'Predict' button is clicked
if st.button('Predict'):
    input_data = {
        'Warehouse_block': warehouse_block,
        'Mode_of_Shipment': mode_of_shipment,
        'Customer_care_calls': customer_care_calls,
        'Customer_rating': customer_rating,
        'Cost_of_the_Product': cost_of_product,
        'Prior_purchases': prior_purchases,
        'Product_importance': product_importance,
        'Gender': gender,
        'Discount_offered': discount_offered,
        'Weight_in_gms': weight_in_gms
    }

    prediction = predict_delivery_on_time_app(log_reg_model, input_data, model_features)

    if prediction == 1:
        st.success('Prediction: The delivery is likely to be ON TIME.')
    else:
        st.error('Prediction: The delivery is likely to be LATE.')
