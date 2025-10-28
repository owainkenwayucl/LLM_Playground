import requests

base_model_endpoint = "http://huggingface-granite.kubeflow-o-kenway.svc.cluster.local"
api_endpoint = f"{base_model_endpoint}/openai/v1/completions"
headers = {
    "Content-Type": "application/json",
}

payload = {
    "model": "granite",
    "prompt": "Tell me some facts about frogs.",
    "max_tokens": 500
}

response = requests.post(api_endpoint, headers=headers, json=payload)

if response.status_code == 200:
    output = response.json()
    print("Generated text:", output["choices"][0]["text"])
else:
    print("Request failed:", response.status_code, response.text)