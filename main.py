# Import necessary libraries
import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np


# Initialize Flask app
app = Flask(__name__)

# Load data and model
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl", "rb"))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())  # Fetch unique locations
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')  # Fix variable name
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    print(location, bhk, bath, sqft)  # Debugging line

    # Create input DataFrame
    input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])

    # Make prediction
    prediction = pipe.predict(input_data)[0] * 1e5

    return str(np.round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True, port=5001)






