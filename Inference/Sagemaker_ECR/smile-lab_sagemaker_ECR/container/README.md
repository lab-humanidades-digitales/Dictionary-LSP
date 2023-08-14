The model have to be in this folder

# Command to convert the model in tar.gz format
This converted file will be used to upload to the correspond AWS S3 Bucket from where torchserve will retrieve the model during the inference process.

Run 
```tar -czvf model.tar.gz model.pt code```

NOTES:   
* Remember to download the .pth file that contains the keypoint estimation model. There is a sh file that use gdown (from pip) to retrieve this model.
* Remember to locate the PSL recognition model (.pt or .pth) in this folder 
