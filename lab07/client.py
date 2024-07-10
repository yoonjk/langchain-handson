import requests

inputs = {"input": "tree"}
response = requests.post("http://localhost:8000/invoke", json=inputs)

print(response.json())