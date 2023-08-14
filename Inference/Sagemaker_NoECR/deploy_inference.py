from sagemaker.pytorch import PyTorchModel
from sagemaker import get_execution_role
from sagemaker.serverless import ServerlessInferenceConfig
#from sagemaker.autoscaling import ScalingPolicy

import os
import json

import boto3


config_file = '../../configuration/config.json'

print("retrieving aws credentials...")
with open(config_file, 'r') as json_file:
    config = json.load(json_file)
    
os.environ['AWS_DEFAULT_REGION'] =  config["credential"]['region']
os.environ['AWS_ACCESS_KEY_ID'] = config["credential"]['id']
os.environ['AWS_SECRET_ACCESS_KEY'] = config["credential"]['key'] 

role = config["credential"]['role']

# You can also configure a sagemaker role and reference it by its name.
# role = "CustomSageMakerRoleName"

print("preparing pytorch model...")
'''
If you don't have model.tar.gz file, please read the readme located in this folder
'''
pytorch_model = PyTorchModel(
    model_data='s3://sagemaker-us-east-1-psl/model.tar.gz', 
    role=role, 
    framework_version="1.12.1",
    py_version="py38",
    #source_dir='./code',
    #source_dir="s3://sagemaker-us-east-1-psl/sourcedir.tar.gz",
    entry_point='inference_keypoints.py')

'''
If all was ok, you will see a charging bar that means the inference is being created. 
You can check it in Sagemaker endpoint dashboard 
'''

print("deploying the model in sagemaker endpoint...")
predictor = pytorch_model.deploy(
    instance_type= 'ml.g4dn.xlarge',#'ml.c6g.large',
    endpoint_name='spoter-Sagemaker-Endpoint-serverless-50c-69a-top5-keypoints',
    serverless_inference_config=ServerlessInferenceConfig(
        max_concurrency=2,
        memory_size_in_mb=1024*3),
    initial_instance_count=1)

'''
# Prepare AutoScaling
asg_client = boto3.client('application-autoscaling') # Common class representing Application Auto Scaling for SageMaker amongst other services
resource_id=f"endpoint/{predictor.endpoint_name}/variant/AllTraffic"
response = asg_client.register_scalable_target(
    ServiceNamespace='sagemaker', #
    ResourceId=resource_id,
    ScalableDimension='sagemaker:variant:DesiredInstanceCount', #"DesiredProvisionedConcurrency"
    MinCapacity=0,
    MaxCapacity=8,
)

print(response)

response = asg_client.put_scaling_policy(
    PolicyName='Invocations-ScalingPolicy',
    ServiceNamespace='sagemaker', # The namespace of the AWS service that provides the resource. 
    ResourceId=resource_id, # Endpoint name 
    ScalableDimension='sagemaker:variant:DesiredInstanceCount', # SageMaker supports only Instance Count
    PolicyType='TargetTrackingScaling', # 'StepScaling'|'TargetTrackingScaling'
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 5.0, # The target value for the metric. 
        'CustomizedMetricSpecification': {
            'MetricName': 'ApproximateBacklogSizePerInstance',
            'Namespace': 'AWS/SageMaker',
            'Dimensions': [
                {'Name': 'EndpointName', 'Value': predictor.endpoint_name }
            ],
            'Statistic': 'Average',
        },
        'ScaleInCooldown': 8, # The cooldown period helps you prevent your Auto Scaling group from launching or terminating 
                                # additional instances before the effects of previous activities are visible. 
                                # You can configure the length of time based on your instance startup time or other application needs.
                                # ScaleInCooldown - The amount of time, in seconds, after a scale in activity completes before another scale in activity can start. 
        'ScaleOutCooldown': 8 # ScaleOutCooldown - The amount of time, in seconds, after a scale out activity completes before another scale out activity can start.
        
        # 'DisableScaleIn': True|False - ndicates whether scale in by the target tracking policy is disabled. 
                            # If the value is true , scale in is disabled and the target tracking policy won't remove capacity from the scalable resource.
    }
)

print(response)
'''