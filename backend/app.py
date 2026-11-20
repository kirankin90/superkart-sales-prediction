
# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize Flask app with a name
superkart_api = Flask("SuperKart")

# Load the trained model
model = joblib.load("superkart_model.joblib")

# Features expected by the model (same order as used in training)
FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]

# Define a route for the home page
@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart System"

# Define an endpoint to predict sales for a single product
@superkart_api.post('/v1/predict')
def predict_sales():
    # Get JSON data from the request
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Request body must be valid JSON'}), 400

    # Reject incomplete requests instead of failing with a server error
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    # Extract relevant features from the input data
    sample = {feature: data[feature] for feature in FEATURES}

    # Convert the extracted data into a DataFrame
    input_data = pd.DataFrame([sample])

    # Make a prediction using the trained model
    prediction = model.predict(input_data).tolist()[0]

    # Return the prediction as a JSON response
    return jsonify({'Sales': round(prediction, 2)})

# Define an endpoint to predict sales for a batch of products
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    # Get the uploaded CSV file from the request
    file = request.files.get('file')
    if file is None:
        return jsonify({'error': "Upload a CSV file using the form field 'file'"}), 400

    # Read the file into a DataFrame
    input_data = pd.read_csv(file)

    missing = [f for f in FEATURES if f not in input_data.columns]
    if missing:
        return jsonify({'error': f'Missing columns: {missing}'}), 400

    # Make predictions for the batch data (columns re-ordered to match training)
    predictions = model.predict(input_data[FEATURES]).tolist()

    # Create an output dictionary mapping row index to predicted sales
    output_dict = {str(i): round(pred, 2) for i, pred in enumerate(predictions)}

    return jsonify(output_dict)


# Run the Flask app in debug mode
if __name__ == '__main__':
    superkart_api.run(debug=True)
