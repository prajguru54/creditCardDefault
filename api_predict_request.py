import requests
import json

url = "http://127.0.0.1:5000/"

payload = json.dumps({
  "name": "Sribastav",
  "email": "prajguru54@gmail.com"
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)