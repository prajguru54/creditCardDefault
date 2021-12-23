from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods = ["GET"])
def home():
    return render_template('index.html')

@app.route('/prediction', methods = ["GET", "POST"])
def predict():
    if request.method == "POST":       
        filepath = request.form.get('filepath')
        return_message =  filepath
        return return_message
    return None

if __name__ == "__main__":
    app.run(debug=True)