

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 01:42:39 2021

@author: Joe
"""

# Standard library imports
import os

# Third party imports
import mediapipe as mp
import numpy as np
import pytorch_lightning as pl
import torchvision
import torch
import math
import cv2

from extract_keypoints import keypointsFormat
from extract_poseflow import read_pose, impute_missing_keypoints, calc_pose_flow
from transforms import Compose, Scale, ToFloatTensor, PermuteImage, Normalize, NORM_STD_IMGNET, \
    NORM_MEAN_IMGNET, CenterCrop, IMAGE_SIZE
import module
#########################
# ARGS
##############

#########################
# MODELS(Mediapipe)
#
# -Holistic
##############

print("\n#####\nHolistic Model\n#####\n")
mp_holistic = mp.solutions.holistic

#########################
# MODELS PARAMETERS
##############

# HOLISTIC parameters.
holistic = mp_holistic.Holistic(static_image_mode=True,
                                model_complexity=2,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)


IdCount = 1
mp4Path = './data/mp4/COMER_915_color.mp4'

# Create a VideoCapture object
cap = cv2.VideoCapture(mp4Path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Check if camera opened successfully
assert cap.isOpened()

idx = 0

w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Init video size:",w_frame, h_frame)

ret, frame = cap.read()

keypointsDict = []

# While a frame was read
while ret is True:

    idx += 1  # Frame count starts at 1

    # temporal variables
    kpDict = {}

    # Convert the BGR image to RGB before processing.
    imageBGR = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = imageBGR.shape





    holisResults = holistic.process(imageBGR)

    # POSE

    kpDict["pose"]={}
    if holisResults.pose_landmarks:

        kpDict["pose"]["x"] = [point.x for point in holisResults.pose_landmarks.landmark]
        kpDict["pose"]["y"] = [point.y for point in holisResults.pose_landmarks.landmark]

    else:
        kpDict["pose"]["x"] = [1.0 for point in range(0, 33)]
        kpDict["pose"]["y"] = [1.0 for point in range(0, 33)]

    # HANDS

    # Left hand
    kpDict["left_hand"]={}
    if(holisResults.left_hand_landmarks):

        kpDict["left_hand"]["x"] = [point.x for point in holisResults.left_hand_landmarks.landmark]
        kpDict["left_hand"]["y"] = [point.y for point in holisResults.left_hand_landmarks.landmark]

    else:
        kpDict["left_hand"]["x"] = [1.0 for point in range(0, 21)]
        kpDict["left_hand"]["y"] = [1.0 for point in range(0, 21)]

    # Right hand
    kpDict["right_hand"]={}
    if(holisResults.right_hand_landmarks):

        kpDict["right_hand"]["x"] = [point.x for point in holisResults.right_hand_landmarks.landmark]
        kpDict["right_hand"]["y"] = [point.y for point in holisResults.right_hand_landmarks.landmark]

    else:
        kpDict["right_hand"]["x"] = [1.0 for point in range(0, 21)]
        kpDict["right_hand"]["y"] = [1.0 for point in range(0, 21)]

    # Face mesh

    kpDict["face"]={}

    if(holisResults.face_landmarks):

        kpDict["face"]["x"] = [point.x for point in holisResults.face_landmarks.landmark]
        kpDict["face"]["y"] = [point.y for point in holisResults.face_landmarks.landmark]

    else:
        kpDict["face"]["x"] = [1.0 for point in range(0, 468)]
        kpDict["face"]["y"] = [1.0 for point in range(0, 468)]

    keypointsDict.append(kpDict)
    #video.write(imageBGR)
    # Next frame
    ret, frame = cap.read()
#########################
# CLOSE MODELS
##############
holistic.close()


height, width, channels = imageBGR.shape
print("N° frames:",idx, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

##############################
# OBTAIN M-DECOSTER KEYPOINTS

newKps = []

for pos, timestep in enumerate(keypointsDict):

    keys = timestep.keys()
    opd = keypointsFormat(timestep, keys)
    newKps.append(opd)

##############################
# OBTAIN POSEFLOW
input_dir_index = 0

# 1. Collect all keypoint files and pre-process them
poses = []
for newKp in newKps:
    poses.append(read_pose(newKp))

poses = np.stack(poses)
poses = impute_missing_keypoints(poses)

# 2. Compute pose flow
prev = poses[0]

flowData = []

for i in range(1, poses.shape[0]):
    next = poses[i]
    flow = calc_pose_flow(prev, next)
    flowData.append(flow)
    prev = next

##############################
num_frames = idx

##### TODO This parameters are "args"
sequence_length = 17
temporal_stride = 1

##############################
# PADDING
frame_start = (num_frames - sequence_length) // (2 * temporal_stride)
frame_end = frame_start + sequence_length * temporal_stride
if frame_start < 0:
    frame_start = 0
if frame_end > num_frames:
    frame_end = num_frames
frame_indices = list(range(frame_start, frame_end, temporal_stride))
while len(frame_indices) < sequence_length:
    # Pad
    frame_indices.append(frame_indices[-1])

##############################
# GET CLIP OF HANDS

SHOULDER_DIST_EPSILON = 1.2
WRIST_DELTA = 0.15

transform = Compose(Scale(IMAGE_SIZE * 8 // 7), CenterCrop(IMAGE_SIZE), ToFloatTensor(),
                            PermuteImage(),
                            Normalize(NORM_MEAN_IMGNET, NORM_STD_IMGNET))

frames, _, _ = torchvision.io.read_video(os.path.join(mp4Path), pts_unit='sec')

clip = []
poseflow_clip = []
missing_wrists_left, missing_wrists_right = [], []

for frame_index in frame_indices:

    value = newKps[frame_index]
    keypoints = np.array(value['people'][0]['pose_keypoints_2d'])
    x = keypoints[0::3]
    y = keypoints[1::3]
    keypoints = np.stack((x, y), axis=0)

    poseflow = None

    frame_index_poseflow = frame_index
    if frame_index_poseflow > 0:

        if len(flowData) <= frame_index_poseflow:
            frame_index_poseflow = len(flowData) - 1

        value = flowData[frame_index_poseflow]

        poseflow = value
        # Normalize the angle between -1 and 1 from -pi and pi
        poseflow[:, 0] /= math.pi
        # Magnitude is already normalized from the pre-processing done before calculating the flow
    else:
        poseflow = np.zeros((33, 2))

    try:
        frame = frames[frame_index]
    except:
        print("ERROR",frame_index,sample['path'])
        continue

    left_wrist_index = 15
    left_elbow_index = 13
    right_wrist_index = 16
    right_elbow_index = 14

    # Crop out both wrists and apply transform
    left_wrist = keypoints[0:2, left_wrist_index]
    left_elbow = keypoints[0:2, left_elbow_index]
    left_hand_center = left_wrist + WRIST_DELTA * (left_wrist - left_elbow)
    left_hand_center_x = left_hand_center[0]
    left_hand_center_y = left_hand_center[1]
    shoulder_dist = np.linalg.norm(keypoints[0:2, 11] - keypoints[0:2, 12]) * SHOULDER_DIST_EPSILON
    left_hand_xmin = max(0, int(left_hand_center_x - shoulder_dist // 2))
    left_hand_xmax = min(frame.size(1), int(left_hand_center_x + shoulder_dist // 2))
    left_hand_ymin = max(0, int(left_hand_center_y - shoulder_dist // 2))
    left_hand_ymax = min(frame.size(0), int(left_hand_center_y + shoulder_dist // 2))

    if not np.any(left_wrist) or not np.any(
            left_elbow) or left_hand_ymax - left_hand_ymin <= 0 or left_hand_xmax - left_hand_xmin <= 0:
        # Wrist or elbow not found -> use entire frame then
        left_hand_crop = frame
        missing_wrists_left.append(len(clip) + 1)
    else:
        left_hand_crop = frame[left_hand_ymin:left_hand_ymax, left_hand_xmin:left_hand_xmax, :]
    left_hand_crop = transform(left_hand_crop.numpy())

    right_wrist = keypoints[0:2, right_wrist_index]
    right_elbow = keypoints[0:2, right_elbow_index]
    right_hand_center = right_wrist + WRIST_DELTA * (right_wrist - right_elbow)
    right_hand_center_x = right_hand_center[0]
    right_hand_center_y = right_hand_center[1]
    right_hand_xmin = max(0, int(right_hand_center_x - shoulder_dist // 2))
    right_hand_xmax = min(frame.size(1), int(right_hand_center_x + shoulder_dist // 2))
    right_hand_ymin = max(0, int(right_hand_center_y - shoulder_dist // 2))
    right_hand_ymax = min(frame.size(0), int(right_hand_center_y + shoulder_dist // 2))

    if not np.any(right_wrist) or not np.any(
            right_elbow) or right_hand_ymax - right_hand_ymin <= 0 or right_hand_xmax - right_hand_xmin <= 0:
        # Wrist or elbow not found -> use entire frame then
        right_hand_crop = frame
        missing_wrists_right.append(len(clip) + 1)
    else:
        right_hand_crop = frame[right_hand_ymin:right_hand_ymax, right_hand_xmin:right_hand_xmax, :]
    right_hand_crop = transform(right_hand_crop.numpy())

    crops = torch.stack((left_hand_crop, right_hand_crop), dim=0)

    clip.append(crops)
    pose_transform = Compose(ToFloatTensor())

    poseflow = pose_transform(poseflow).view(-1)
    poseflow_clip.append(poseflow)

# Try to impute hand crops from frames where the elbow and wrist weren't missing as close as possible temporally
for clip_index in range(len(clip)):
    if clip_index in missing_wrists_left:
        # Find temporally closest not missing frame for left wrist
        replacement_index = -1
        distance = np.inf
        for ci in range(len(clip)):
            if ci not in missing_wrists_left:
                dist = abs(ci - clip_index)
                if dist < distance:
                    distance = dist
                    replacement_index = ci
        if replacement_index != -1:
            clip[clip_index][0] = clip[replacement_index][0]
    # Same for right crop
    if clip_index in missing_wrists_right:
        # Find temporally closest not missing frame for right wrist
        replacement_index = -1
        distance = np.inf
        for ci in range(len(clip)):
            if ci not in missing_wrists_right:
                dist = abs(ci - clip_index)
                if dist < distance:
                    distance = dist
                    replacement_index = ci
        if replacement_index != -1:
            clip[clip_index][1] = clip[replacement_index][1]

clip = torch.stack(clip, dim=0)
poseflow_clip = torch.stack(poseflow_clip, dim=0)

x = (clip, poseflow_clip)

#TODO change path
chk_path = "./checkpoints/bestLoggedModel.ckpt"

model = module.get_model_def().load_from_checkpoint(chk_path)
trainer = pl.Trainer()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.to(device)
model.eval()

#to simulate a minibatch
clip = clip.unsqueeze(0)
poseflow_clip = poseflow_clip.unsqueeze(0)

with torch.no_grad():

    if isinstance(x, list):

        logits = model([e.to(device) for e in x]).cpu()
    else:
        logits = model([clip.to(device),poseflow_clip.to(device)]).cpu()

    predictions = torch.argmax(logits, dim=1)

    print()
    print(f"prediction's key is: {int(predictions)}")
    #print(predictions)