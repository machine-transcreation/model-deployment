import requests

url = 'https://api.runpod.ai/v2/skgg94h4lchsy0/runsync'
headers = {
    'Authorization': 'Bearer 3N2FXXVFVBW6ANKM5FRIFVWHP1701AN3TYEW3CAC',
    'Content-Type': 'application/json'
}

with open('base_test.json', 'r') as file:
    payload = file.read()

response = requests.post(url, headers=headers, data=payload)

print(response.status_code)
print(response.json())
