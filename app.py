from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import joblib

app = Flask(__name__)

# Load your trained model and scaler
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Create input array for the model
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Scale the input data
        scaled_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_data)[0]

        # Return the prediction as JSON
        return jsonify({'prediction': prediction})

    except Exception as e:
        print("Error:", e)
        return str(e)

@app.route('/result')
def result():
    prediction = request.args.get('prediction', 'No prediction')
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
