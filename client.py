import requests
import json

# API Endpoint
API_URL = "http://127.0.0.1:8000/evaluate/"

# Example Data
payload = {
    "forget_data": [
        {"question": "What is Alice's favourite colour?", "answer": "green"},
        {"question": "Where does Alice live?", "answer": "Alice lives in the UK."}
    ],
    "retain_data": [
        {"question": "What is the capital of France?", "answer": "The capital of France is Paris."},
        {"question": "Who wrote Hamlet?", "answer": "William Shakespeare wrote Hamlet."}
    ]
}

# Send POST request to FastAPI server
response = requests.post(API_URL, json=payload)

# Check if request was successful
if response.status_code == 200:
    result = response.json()
    print("✅ API Response:")
    print(json.dumps(result, indent=4))
else:
    print(f"❌ Error: {response.status_code}, {response.text}")
