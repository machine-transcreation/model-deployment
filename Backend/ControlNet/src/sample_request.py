import requests
import json

json_file_path = 'sample_input.json'
txt_file_path = 'response.txt'


with open(json_file_path, 'r') as file:
    json_data = json.load(file)


url = 'https://api.runpod.ai/v2/vbiz2qzgkfnngk/run'

headers = {"Content-Type": "application/json", "Authorization": "Bearer vbiz2qzgkfnngk"}

response = requests.post(url, json=json_data, headers=headers)


if response.status_code == 200:

    with open(txt_file_path, 'w') as file:
        file.write(response.text)
    print(f"Response has been written to {txt_file_path}")
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")
    print(f"Response: {response.text}")