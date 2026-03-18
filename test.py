import requests

url = "https://clinicaltrials.gov/api/v2/studies"
params = {
    "query.term": "lung cancer pembrolizumab PHASE2",
    "pageSize": 3,
    "format": "json",
}

r = requests.get(url, params=params, timeout=30)
print(r.status_code)
print(r.url)
print(r.text[:500])