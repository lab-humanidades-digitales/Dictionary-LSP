from sagemaker.pytorch import PyTorchModel
from sagemaker import get_execution_role
import os
import json

config_file = '../../../configuration/config.json'

with open(config_file, 'r') as json_file:
    config = json.load(json_file)
    
os.environ['AWS_DEFAULT_REGION'] =  config["credential"]['region']
os.environ['AWS_ACCESS_KEY_ID'] = config["credential"]['id']
os.environ['AWS_SECRET_ACCESS_KEY'] = config["credential"]['key'] 

role = config["credential"]['role']

# You can also configure a sagemaker role and reference it by its name.
# role = "CustomSageMakerRoleName"

pytorch_model = PyTorchModel(
    model_data='', 
    role=role, 
    entry_point='./inference.py', 
    image_uri='',
    framework_version='1.3.1')

predictor = pytorch_model.deploy(
    instance_type='ml.t2.medium',
    endpoint_name='Sagemaker-Endpoint-model',
    initial_instance_count=1)