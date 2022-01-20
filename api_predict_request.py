import requests
import json

url = "http://127.0.0.1:5000/predict_json"
prediction_filepath = 'Prediction_Batch_files'
#prediction_filepath = 'Prediction1234' --> This is invalid filepath for testing failure condition

payload = json.dumps({"filepath": prediction_filepath})
headers = {'Content-Type': 'application/json'}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)