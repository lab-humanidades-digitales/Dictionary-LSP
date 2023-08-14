import boto3
import os
import json


config_file = '../../../../configuration/config.json'

with open(config_file, 'r') as json_file:
    config = json.load(json_file)
    
os.environ['AWS_DEFAULT_REGION'] =    config["credential"]['region']
os.environ['AWS_ACCESS_KEY_ID'] =     config["credential"]['id']
os.environ['AWS_SECRET_ACCESS_KEY'] = config["credential"]['key'] 
 
endpoint = 'Sagemaker-Endpoint-model'
 
runtime = boto3.Session().client('sagemaker-runtime')
 
# Read image into memory
payload = 'from invoke.py (repository)'
# Send image via InvokeEndpoint API
response = runtime.invoke_endpoint(EndpointName=endpoint, ContentType='application/json', Body=payload)

# Unpack response
result = json.loads(response['Body'].read().decode())
print(result)