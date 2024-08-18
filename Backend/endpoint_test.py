import os
from dotenv import load_dotenv, dotenv_values
import requests

load_dotenv(".env")

url = os.getenv("PAINT_ENDPOINT")
headers = {
    'Authorization': f'Bearer {os.getenv("RUNPOD_KEY")}',
    'Content-Type': 'application/json'
}

with open('Paint-by-Example/src/base_test.json', 'r') as file:
    payload = file.read()

response = requests.post(url, headers=headers, data=payload)

print(response.status_code)
print(response.json())
