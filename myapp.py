from flask import Flask, request, render_template
from prediction import predict_all_files

app = Flask(__name__)

@app.route('/', methods = ["GET"])
def home():
    return render_template('index.html')

#Use this route to send prediction request from frontend(web interface)
@app.route('/prediction', methods = ["GET", "POST"])
def predict():
    if request.method == "POST":       
        filepath = request.form.get('hero-field')
        prediction_output_path = predict_all_files(filepath)
        return render_template('result.html', prediction_output_path = prediction_output_path)
    return None

#Use this route to send prediction request from static programme like python, postman etc.
@app.route("/predict_json", methods = [ "POST"])
def predict_json():
    if request.method == "POST":
        req_data = request.get_json()
        filepath = req_data.get('filepath')
        prediction_output_path = predict_all_files(filepath)
        return f'Prediction files are created at {prediction_output_path}'



if __name__ == "__main__":
    app.run(debug=True)