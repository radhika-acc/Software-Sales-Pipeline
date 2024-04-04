import os
import subprocess

# Run the Jupyter notebook script
subprocess.run(['jupyter', 'nbconvert', '--to', 'script', 'Term Project.ipynb'])


from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('opportunity_to_sales_model.keras')

model_opp = tf.keras.models.load_model('Lead_to_Opportunity_model.keras')

# Load the scaler
scaler = StandardScaler()
scaler.mean_ = np.load('scaler_mean.npy')
scaler.scale_ = np.load('scaler_scale.npy')

# Define categorical columns
categorical_columns = ['Authority', 'Comp_size', 'Competitors', 'Purch_dept', 'Partnership', 
                       'Budgt_alloc', 'Forml_tend', 'RFI', 'RFP', 'Growth', 'Posit_statm', 
                       'Client', 'Scope', 'Strat_deal', 'Cross_sale', 'Deal_type', 'Needs_def', 
                       'Att_t_client', 'Source', 'Up_sale','Opportunity']

# Define label encoders for categorical variables
label_encoders = {col: LabelEncoder() for col in categorical_columns}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from form data
    features = request.form.to_dict()

    # Determine which button was clicked
    prediction_type = request.form['prediction_type']
    
    # Fit label encoders with form data
    fit_label_encoders(features)

    # Preprocess input features
    features_encoded = preprocess_input(features)
    features_scaled = scaler.transform(features_encoded)
    
    # Make prediction based on the button clicked
    if prediction_type == 'Predict Sales':
        prediction = model.predict(features_scaled)
        prediction_text = "Probability of closing this deal"
    elif prediction_type == 'Predict Opportunity':
        prediction = model_opp.predict(features_scaled)
        prediction_text = "Probability of getting this client"
    
    
    return render_template('result.html', prediction=prediction[0][0], prediction_text=prediction_text)
    
    
def fit_label_encoders(features):
    # Get all feature values for each column
    all_values = {col: [] for col in features.keys()}
    for col, value in features.items():
        all_values[col].append(value)
        

    # Fit label encoders with all feature values
    for col, values in all_values.items():
        if col in label_encoders:
            label_encoders[col].fit(values)

def preprocess_input(features):
    # Convert categorical values to numerical labels
    label_encoded_features = {}
    for col, encoder in label_encoders.items():
        label_encoded_features[col] = encoder.transform([features[col]])[0]
    features_encoded = pd.DataFrame([label_encoded_features])
    return features_encoded

if __name__ == '__main__':
    app.run(debug=True)
