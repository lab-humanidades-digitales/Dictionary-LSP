import mediapipe as mp
import pandas as pd
import numpy as np
import cv2 
import torch
import json
from collections import OrderedDict

from spoter.spoter_model import SPOTER

HIDDEN_DIM = {
    "29": 58,
    "51": 102,
    "71": 142
}

NUM_CLASSES = {
    "AEC": 28,
    "PUCP_PSL_DGI156": 29,
    "WLASL": 86,
    "AEC-DGI156-DGI305":72
}

def get_mp_keys(points):
    tar = np.array(points.mp_pos)-1
    return list(tar)

def get_op_keys(points):
    tar = np.array(points.op_pos)-1
    return list(tar)

def get_wp_keys(points):
    tar = np.array(points.wb_pos)-1
    return list(tar)

def parse_configuration(config_file):
    """Loads config file if a string was passed
        and returns the input if a dictionary was passed.
    """
    if isinstance(config_file, str):
        with open(config_file, 'r') as json_file:
            return json.load(json_file)
    else:
        return config_file

def configure_model(config_file, use_wandb):

    config_file = parse_configuration(config_file)

    config = dict(
        #hidden_dim = config_file["hparams"]["hidden_dim"],
        #num_classes = config_file["hparams"]["num_classes"],
        epochs = config_file["hparams"]["epochs"],
        num_backups = config_file["hparams"]["num_backups"],
        keypoints_model = config_file["hparams"]["keypoints_model"],
        lr = config_file["hparams"]["lr"],
        keypoints_number = config_file["hparams"]["keypoints_number"],

        nhead = config_file["hparams"]["nhead"],
        num_encoder_layers = config_file["hparams"]["num_encoder_layers"],
        num_decoder_layers = config_file["hparams"]["num_decoder_layers"],
        dim_feedforward = config_file["hparams"]["dim_feedforward"],

        experimental_train_split = config_file["hparams"]["experimental_train_split"],
        validation_set = config_file["hparams"]["validation_set"],
        validation_set_size = config_file["hparams"]["validation_set_size"],
        log_freq = config_file["hparams"]["log_freq"],
        save_checkpoints = config_file["hparams"]["save_checkpoints"],
        scheduler_factor = config_file["hparams"]["scheduler_factor"],
        scheduler_patience = config_file["hparams"]["scheduler_patience"],
        gaussian_mean = config_file["hparams"]["gaussian_mean"],
        gaussian_std = config_file["hparams"]["gaussian_std"],
        plot_stats = config_file["hparams"]["plot_stats"],
        plot_lr = config_file["hparams"]["plot_lr"],

        #training_set_path = config_file["data"]["training_set_path"],
        #validation_set_path = config_file["data"]["validation_set_path"],
        #testing_set_path = config_file["data"]["testing_set_path"],

        n_seed = config_file["seed"],
        device = config_file["device"],
        dataset_path = config_file["dataset_path"],
        weights_trained = config_file["weights_trained"],
        save_weights_path = config_file["save_weights_path"],
        dataset = config_file["dataset"]
    )
    return config

def frame_process(holistic, frame):

    imageBGR = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = imageBGR.shape

    holisResults = holistic.process(imageBGR)

    kpDict = {}

    #POSE

    if holisResults.pose_landmarks:
        pose = [ [point.x, point.y] for point in holisResults.pose_landmarks.landmark]
    else:
        pose = [ [0.0, 0.0] for point in range(0, 33)]
    pose = np.asarray(pose)

    # HANDS

    # Left hand
    if(holisResults.left_hand_landmarks):
        left_hand = [ [point.x, point.y] for point in holisResults.left_hand_landmarks.landmark]
    else:
        left_hand = [ [0.0, 0.0] for point in range(0, 21)]
    left_hand = np.asarray(left_hand)

    # Right hand
    if(holisResults.right_hand_landmarks):
        right_hand = [ [point.x, point.y] for point in holisResults.right_hand_landmarks.landmark]

    else:
        right_hand = [ [0.0, 0.0] for point in range(0, 21)]
    right_hand = np.asarray(right_hand)

    # Face mesh

    if(holisResults.face_landmarks):

        face = [ [point.x, point.y] for point in holisResults.face_landmarks.landmark]

    else:
        face = [[0.0, 0.0] for point in range(0, 468)]
    face = np.asarray(face)

    newFormat = []

    newFormat.append(pose)
    newFormat.append(face)
    newFormat.append(left_hand)
    newFormat.append(right_hand)

    x = np.asarray([item[0] for sublist in newFormat for item in sublist])
    y = np.asarray([item[1] for sublist in newFormat for item in sublist])

    data = np.asarray([x,y])

    return data


def preprocess_video():
    model_key_getter = {'mediapipe': get_mp_keys,
                    'openpose': get_op_keys,
                    'wholepose': get_wp_keys}

    model_key_getter = model_key_getter['mediapipe']

    # 71 points
    num_joints = 29
    points = pd.read_csv(f"./points_{num_joints}.csv")

    input_path = 'Data/mp4/casa_1418.mp4'
    path = input_path

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic()

    cap = cv2.VideoCapture(path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    print(path)

    output_list = []
    
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break 

        pred = frame_process(holistic, img)

        selected_joints = model_key_getter(points)
        
        pred = pred[:,selected_joints]

        output_list.append(pred)

        success, img = cap.read()

    result = np.asarray(output_list)

    return result

def preprocess_keypoints(data):
    depth_map = torch.from_numpy(np.copy(data))
    depth_map = depth_map - 0.5

    return depth_map

def preproccess():
    data = preprocess_video()
    data = preprocess_keypoints(data)

    return data

def load_model(path):

    CONFIG_FILENAME = "config.json"

    config = configure_model(CONFIG_FILENAME, False)

    model = SPOTER(num_classes=NUM_CLASSES[config['dataset']], 
                        hidden_dim=HIDDEN_DIM[str(config['keypoints_number'])],
                        dim_feedforward=config['dim_feedforward'],
                        num_encoder_layers=config['num_encoder_layers'],
                        num_decoder_layers=config['num_decoder_layers'],
                        nhead=config['nhead']
                        )

    if '.pkl' in path:
        with open(path, 'r') as f:
            weights = pickle.load(f)
    else:
        weights = torch.load(path, map_location=torch.device('cpu'))


    try:
        model.load_state_dict(weights)
    except:
        state = model.state_dict()
        diff = list(set(state.keys()).difference(set(weights.keys())))
        print('Can not find these weights:')
        for d in diff:
            print('  ' + d)
        state.update(weights)
        model.load_state_dict(state)

    return model


#####################################################################################

def input_fn(request_body, request_content_type):
    data = preproccess()

    #data = torch.Tensor(data)
    #data = Variable(data.float().cpu(), requires_grad=False)

    return data #"input"#data.transpose(3,1).transpose(2,3)

def model_fn(model_dir):
    
    model = load_model(model_dir)
   
    #model = load_model(arg)

    return model #"model" 

def predict_fn(input_data, model):

    with torch.no_grad():
        output = model(input_data).expand(1, -1, -1)
    return output



def output_fn(prediction, content_type):
    
    _, predict_label = torch.topk(prediction.data, 5)

    result = predict_label[0][0].numpy() 

    with open('meaning.json') as f:
        meaning = json.load(f)

    result = [meaning[str(val)] for val in result]

    return result

data = input_fn("","")
model = model_fn('../model.pth')
pred = predict_fn(data, model)
print(output_fn(pred, ""))