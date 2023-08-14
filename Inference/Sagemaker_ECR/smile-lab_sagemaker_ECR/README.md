NOTE: Remeber to modify (and not upload to git) the config.json file located in configuration folder in the main root in order to use in deploy.py your aws credential.

# Steps to prepare the use of the inference

1. prepare the config.json file located in configuration folder in the main folder.

2. Place the smile-lab model in "container" folder and call it "model" (.pt or .pth)

3.  You can use "download_kp_model.sh" file (in container/code) or this  [link](https://drive.google.com/file/d/1f_c3uKTDQ4DR3CrwMSI8qdsTKJvKVt7p/view) to download the Hrnet pre-trained model for keypoint estimation. this file have to be located in "code"

## In case you want to try the inference locally
1. use the yaml file placed in "container/code" called "environment.yml" to install all the dependencies the inference need in Conda. 

# Files used to as part of the DockerFile creation process
The following bullet list contains more details about why some files are used to prepare the use of the inference. It also detailed in the same way the files that are inside some folders

* config.properties: It have the configuration of torchserve service
* deep_learning_container.py: It helps to manage the model pointed in a S3 bucket
* Dockerfile: File to create the Docker image
* Invoke.py: Its complete version is in AWS lamda also called as "invoke.py"
* torchserve-entrypoint.py: [and torchserve-ec2-entrypoint.py] this both files are used by otrchserve to manage sagemaker function. We do not know which one we have to use.
* container folder: It is used to copy all the "smile-lab" scripts and model at once to the docker image.

## Folder: container
* deploy.py: used to initialize the sagemaker endpoint
* local_test.py: It is used to try locally the inference. At the same time, this file have another version of "input_fn, model_fn, predict_fn, output_fn" but here it is called as "preprocess, Inference, postprocess". We saved it in case this one is the one we have to use in the inference (in this case, it have to be placed in code folder).
* model.pt: the pre-trained PSL recognition model. If it is not here, you have to train a new one or ask us to share you one.
* code folder: It have all the scripts to make the inference work

## Folder: container/code 
* download_kp_model.sh: file used to download the keypoint estimation model
* inference.py: file that have the structure "input_fn, model_fn, predict_fn, output_fn" to create the inference. It also include the preprocessing part that "smile-lab" model use to prepare the data
* meaning.pkl: dictionary that have the meaning of "smile-lab" model output label.
* model_handler.py: part of torchserve files that helps in the manage of the model
* other files and folder: There are part of "smile-lab" model script tha allow this model works. 

## Folder: container/code/config
* test_joint.yaml: contain the hyperparameters, paths and model options used during the inference process. Remember to set there the same hyperparameters of the pre-trained model you want to use. 

# Links that helped us to develop our scripts

* [how to deploy a pytorch model on Sagemaker](https://samuelabiodun.medium.com/how-to-deploy-a-pytorch-model-on-sagemaker-aa9a38a277b6)
* [Github: Multi-model template](https://github.com/aws/amazon-sagemaker-examples/tree/main/advanced_functionality/multi_model_pytorch)
* [Github: Sagemaker pytorch inference toolkit](https://github.com/aws/sagemaker-pytorch-inference-toolkit)
* [Github: Sagemaker inference toolkit](https://github.com/aws/sagemaker-inference-toolkit)