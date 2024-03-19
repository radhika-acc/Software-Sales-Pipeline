from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from sqlalchemy import create_engine

app = Flask(__name__)

# Load the pre-trained model
model = load_model('opportunity_to_sales_model.keras')  # Assuming you have saved your trained model as 'opportunity_to_sales_model.h5'

# Load label encoder
label_encoder = LabelEncoder()

# Load the dataset
data = pd.read_excel('Sales Dataset.xlsx')
data.columns = data.columns.str.lower().str.replace(' ', '_')
data.rename(columns={'technology\nprimary': 'technology_primary'}, inplace=True)
data.rename(columns={'opportunity_size_(usd)': 'opportunity_size_usd'}, inplace=True)

# Generate new opportunity IDs
data['opportunity_id'] = ['N' + str(i).zfill(8) for i in range(1, len(data) + 1)]

# Mark opportunity_status as 'Unknown'
data['opportunity_status'] = None

# Fit label encoder on entire dataset
for column in data.select_dtypes(include=['object']).columns:
    label_encoder.fit(data[column])

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for processing the form submission
@app.route('/predict', methods=['POST'])
def predict():
    
    # Get user input from the form
    input_data = {}
    for column in data.columns[1:]:
        input_data[column] = [request.form[column]]
    
    # Preprocess the input data
    input_df = pd.DataFrame(input_data)
    
    # Encode categorical variables
    for column in input_df.select_dtypes(include=['object']).columns:
        # Handle unseen labels by replacing them with a default value
        input_df[column] = label_encoder.transform(input_df[column])
    
    # Make prediction
    probability = model.predict(input_df)[0][0]

    return render_template('result.html', probability=probability)

if __name__ == '__main__':
    app.run(debug=True)
