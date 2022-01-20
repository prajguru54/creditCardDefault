from flask import Flask, request, render_template
from prediction import predict_all_files
import os

app = Flask(__name__)

@app.route('/', methods = ["GET"])
def home():
    return render_template('index.html')

#Use this route to send prediction request from frontend(web interface)
@app.route('/prediction', methods = ["GET", "POST"])
def predict():
    if request.method == "POST":       
        filepath = request.form.get('filepath')
        print(f"Filepath : {filepath}")
        if os.path.isdir(filepath):
            print("Valid filepath")
            prediction_output_path = predict_all_files(filepath)
            return render_template('result.html', prediction_output_path = prediction_output_path)
        else:
            print("Invalid filepath")
            return render_template('failure.html', input_filepath = filepath)

#Use this route to send prediction request(in JSON format) from static programme like python, postman etc.
@app.route("/predict_json", methods = [ "POST"])
def predict_json():
    if request.method == "POST":
        req_data = request.get_json()
        filepath = req_data.get('filepath')
        if os.path.isdir(filepath):
            print("Filepath valid")
            prediction_output_path = predict_all_files(filepath)
            return f'Prediction files are created at {prediction_output_path}'
        else:
            print("Invalid filepath")
            return "Please enter a valid filepath"



if __name__ == "__main__":
    app.run(debug=True)