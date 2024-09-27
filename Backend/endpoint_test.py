import os
from dotenv import load_dotenv
import requests

load_dotenv(".env")

url = os.getenv("POWERPAINT_ENDPOINT")
headers = {
    'Authorization': f'Bearer {os.getenv("RUNPOD_KEY")}',
    'Content-Type': 'application/json'
}

with open('testing/powerpaint.json', 'r') as file:
    payload = file.read()

response = requests.post(url, headers=headers, data=payload)

print(response.status_code)
print(response.json())
