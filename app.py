from flask import Flask, render_template, request, jsonify, url_for, redirect
import joblib
import numpy as np

# Load the saved model at the beginning of the script
loaded_model = joblib.load('model.pkl')

# Create a Flask app and define an API route to handle POST requests
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
   # data = request.get_json()
    # Process the data and perform inference using loaded_model.predict()
    # Example: 
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
   # prediction=loaded_model.predict_proba(final)

    result = loaded_model.predict(final)
    #return jsonify({'prediction': result})
    return render_template('index.html', prediction_result=result)

if __name__ == '__main__':
    app.run(debug=True)
