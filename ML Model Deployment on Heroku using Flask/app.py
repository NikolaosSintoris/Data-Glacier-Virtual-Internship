from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd

# Create the instance of the Flask()
app = Flask(__name__)

# Load the model
load_fd = open("model.pkl","rb")# open the file for reading
model = pickle.load(load_fd) # load the object from the file into new_model

# Maps the method defined below, to the URL mentioned inside the decorator.
# index() method would be called automatically, and the index() method returns our main HTML page called index.html
# flask.render_template() looks for the index.html file in the templates folder and dynamically generates an HTML page for the user.
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    features_array = [np.array(features)]
    prediction = model.predict(features_array)[0]
    if prediction == 0:
        outcome = 'No diabetes'
    else:
        outcome = 'Diabetes'
    return render_template('index.html', prediction_text = 'The predicted diagnosis is: {}'.format(outcome))

# API endpoint
@app.route('/predict_api')
def predict():
    api_load_fd = open("model.pkl", "rb")  # open the file for reading
    api_model = pickle.load(api_load_fd)  # load the object from the file into new_model
    pregnancies = request.args.get('pregnancies')
    glucose = request.args.get('glucose')
    blood_press = request.args.get('blood_press')
    skin_thick = request.args.get('skin_thick')
    insulin = request.args.get('insulin')
    bmi = request.args.get('bmi')
    d_pedigree_f = request.args.get('d_pedigree_f')
    age = request.args.get('age')

    api_data_frame = pd.DataFrame(
        {
            'pregnancies': [pregnancies], 'glucose': [glucose], 'blood_press': [blood_press],
            'skin_thick': [skin_thick], 'insulin':[insulin], 'bmi':[bmi], 'd_pedigree_f':[d_pedigree_f], 'age':[age]
        }
    )

    api_prediction = api_model.predict(api_data_frame)[0]
    if api_prediction == 0:
        outcome = 'No diabetes'
    else:
        outcome = 'Diabetes'

    return jsonify({'prediction':outcome})

if __name__ == "__main__":
    app.run(debug=True)