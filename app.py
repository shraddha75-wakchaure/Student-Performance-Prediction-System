from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        studytime = float(request.form['studytime'])
        absences = float(request.form['absences'])
        g1 = float(request.form['g1'])
        g2 = float(request.form['g2'])

        features = np.array([[studytime, absences, g1, g2]])
        prediction = model.predict(features)

        marks = prediction[0]

        # Pass/Fail condition
        if marks >= 10:
            status = "PASS"
            color = "green"
        else:
            status = "FAIL"
            color = "red"

        return render_template('index.html',
                               prediction_text=f"{marks:.2f}",
                               result_status=status,
                               color=color)

    except:
        return render_template('index.html',
                               prediction_text="Invalid Input",
                               result_status="",
                               color="black")

if __name__ == "__main__":
    app.run(debug=True)