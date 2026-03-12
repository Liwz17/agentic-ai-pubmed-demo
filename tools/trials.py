import requests

CLINICALTRIAL_URL = "https://clinicaltrials.gov/api/query/studies"

def search_trials(query):

    params = {
        "query.term": query,
        "format": "json"
    }

    r = requests.get(CLINICALTRIAL_URL, params=params)

    return r.json()