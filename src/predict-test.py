import requests


url = 'http://localhost:9696/predict'


employee = {
    "age": 33.0,
    "gender": "female",
    "education_level": "master's",
    "job_title": "sales_manager",
    "years_of_experience": 10.0
}


response =  requests.post(url,json = employee).json()

print("the expected salary is ",response['salary'])

