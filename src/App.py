# Flask application for model serving
from flask import Flask, request, jsonify
import pickle


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Load the model
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Parse input JSON
    input_data = request.json['data']

    # Perform prediction
    predictions = model.predict(input_data).tolist()

    # Return predictions as JSON
    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")