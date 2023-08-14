from __future__ import print_function

import glob
import json
import logging
import re
from collections import namedtuple

import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as f
from torch.autograd import Variable
import torchvision.transforms as transforms

from collections import OrderedDict
import numpy as np
import pandas as pd
import cv2
import pickle
import yaml

from pose_hrnet import get_pose_net
from utils import pose_process
from config import cfg

import warnings
warnings.filterwarnings("ignore")

def get_mp_keys(points):
    tar = np.array(points.mp_pos)-1
    return list(tar)

def get_op_keys(points):
    tar = np.array(points.op_pos)-1
    return list(tar)

def get_wp_keys(points):
    tar = np.array(points.wb_pos)-1
    return list(tar)

def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(description='Decoupling Graph Convolution Network with DropGraph Module')
    parser.add_argument('--config',default='./config/test_joint.yaml', help='path to the configuration file')
    return parser

def norm_numpy_totensor(img):

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    img = img.astype(np.float32) / 255.0
    for i in range(3):
        img[:, :, :, i] = (img[:, :, :, i] - mean[i]) / std[i]
    return torch.from_numpy(img).permute(0, 3, 1, 2)

def stack_flip(img):
    img_flip = cv2.flip(img, 1)
    return np.stack([img, img_flip], axis=0)

def merge_hm(hms_list):

    index_mirror = np.concatenate([
                [1,3,2,5,4,7,6,9,8,11,10,13,12,15,14,17,16],
                [21,22,23,18,19,20],
                np.arange(40,23,-1), np.arange(50,40,-1),
                np.arange(51,55), np.arange(59,54,-1),
                [69,68,67,66,71,70], [63,62,61,60,65,64],
                np.arange(78,71,-1), np.arange(83,78,-1),
                [88,87,86,85,84,91,90,89],
                np.arange(113,134), np.arange(92,113)]) - 1

    assert isinstance(hms_list, list)
    for hms in hms_list:
        hms[1,:,:,:] = torch.flip(hms[1,index_mirror,:,:], [2])
    
    hm = torch.cat(hms_list, dim=0)
    # print(hm.size(0))
    hm = torch.mean(hms, dim=0)
    return hm

def preproccess():

    model_key_getter = {'mediapipe': get_mp_keys,
                    'openpose': get_op_keys,
                    'wholepose': get_wp_keys}

    model_key_getter = model_key_getter['wholepose']

    # 71 points
    selected_joints =  np.concatenate(([0,5,6,7,8,9,10],
                                    [91,95,96,99,100,103,104,107,108,111],
                                    [112,116,117,120,121,124,125,128,129,132]), axis=0)
    num_joints = 71

    points = pd.read_csv(f"./points_{num_joints}.csv")

    max_body_true = 1
    max_frame = 150
    num_channels = 2

    multi_scales = [512,640]

    with torch.no_grad():
        config = 'wholebody_w48_384x288.yaml'
        cfg.merge_from_file(config)

        # dump_input = torch.randn(1, 3, 256, 256)
        # newmodel = PoseHighResolutionNet()
        newmodel = get_pose_net(cfg, is_train=False)
        #print(newmodel)
        # dump_output = newmodel(dump_input)
        # print(dump_output.size())
        checkpoint = torch.load('./hrnet_w48_coco_wholebody_384x288-6e061c6a_20200922.pth')
        # newmodel.load_state_dict(checkpoint['state_dict'])


        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'backbone.' in k:
                name = k[9:] # remove module.
            if 'keypoint_head.' in k:
                name = k[14:] # remove module.
            new_state_dict[name] = v
        newmodel.load_state_dict(new_state_dict)

        newmodel.cpu().eval()

        transform  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        input_path = 'data/mp4/casa_1418.mp4'
        path = input_path

        step = 600
        start_step = 6

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
            img = cv2.resize(img, (256,256))
            frame_height, frame_width = img.shape[:2]
            img = cv2.flip(img, flipCode=1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out = []
            for scale in multi_scales:
                if scale != 512:
                    img_temp = cv2.resize(img, (scale,scale))
                else:
                    img_temp = img
                img_temp = stack_flip(img_temp)
                img_temp = norm_numpy_totensor(img_temp).cpu()
                hms = newmodel(img_temp)
                if scale != 512:
                    out.append(f.interpolate(hms, (frame_width // 4,frame_height // 4), mode='bilinear'))
                else:
                    out.append(hms)

            out = merge_hm(out)

            result = out.reshape((133,-1))
            result = torch.argmax(result, dim=1)

            result = result.cpu().numpy().squeeze()

            y = result // (frame_width // 4)
            x = result % (frame_width // 4)
            pred = np.zeros((133, 3), dtype=np.float32)
            pred[:, 0] = x
            pred[:, 1] = y

            hm = out.cpu().numpy().reshape((133, frame_height//4, frame_height//4))

            pred = pose_process(pred, hm)
            pred[:,:2] *= 4.0

            selected_joints = model_key_getter(points)

            pred = pred[selected_joints,:2]

            assert pred.shape == (71, 2)

            output_list.append(pred)

            cap.release()

        skel = np.asarray(output_list)

        fp = np.zeros((1, max_frame, num_joints, num_channels, max_body_true), dtype=np.float32)

        if skel.shape[0] < max_frame:
            L = skel.shape[0] 

            fp[0,:L,:,:,0] = skel

            rest = max_frame - L
            num = int(np.ceil(rest / L))
            pad = np.concatenate([skel for _ in range(num)], 0)[:rest]
            fp[0,L:,:,:,0] = pad
        else:
            L = skel.shape[0]
            #print(L)
            fp[0,:,:,:,0] = skel[:max_frame,:,:]

        return fp

def init_seed(_):
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cpu.enabled = False


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def load_model(arg):
    output_device =  arg.device[0] if type(
            arg.device) is list else self.arg.device
    output_device =  output_device
    Model = import_class(arg.model)
    model = Model(**arg.model_args).cpu()
    # print(self.model)

    loss = nn.CrossEntropyLoss().cpu()
    # self.loss = LabelSmoothingCrossEntropy().cuda(output_device)

    if arg.weights:

        if '.pkl' in arg.weights:
            with open(arg.weights, 'r') as f:
                weights = pickle.load(f)
        else:
            weights = torch.load(arg.weights)

        weights = OrderedDict(
            [[k.split('module.')[-1],
                v.cpu()] for k, v in weights.items()])

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

    if type(arg.device) is list:
        if len(arg.device) > 1:
            model = nn.DataParallel(
                model,
                device_ids=arg.device,
                output_device=output_device)
    return model

def preprocess():
    """
    Transform raw input into model input data.
    :param request: list of raw requests
    :return: list of preprocessed model input data
    """

    #if content_type != 'application/json':
    #    raise Exception(f'Requested unsupported ContentType in content_type {content_type}')

    #input_data = json.loads(request_body)
    #url = input_data['url']
    #logger.info(f'Image url: {url}')
    #image_data = Image.open(requests.get(url, stream=True).raw)

    # Take the input data and pre-process it make it inference ready
    data = preproccess()

    data = torch.Tensor(data)
    data = Variable(data.float().cpu(), requires_grad=False)

    return data.transpose(3,1).transpose(2,3)

def inference(input_data):
    """
    Internal inference methods
    :param model_input: transformed model input data list
    :return: list of inference output in NDArray
    """
    parser = get_parser()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            #default_arg = yaml.load(f)
            default_arg = yaml.safe_load(f)

        parser.set_defaults(**default_arg)
    arg = parser.parse_args()

    model = load_model(arg)

    with torch.no_grad():
        output = model(input_data)

    return output

def postprocess(prediction):
    """
    Return predict result in as list.
    :param inference_output: list of inference output
    :return: list of predict results
    """
    # Take output from network and post-process to desired format
    _, predict_label = torch.topk(prediction.data, 5)
            
    result = predict_label[0].numpy() 

    meaning = pd.read_pickle("meaning.pkl")
    meaning = dict(meaning)

    result = [meaning[val] for val in result]

    return result

model_input = preprocess()
model_out = inference(model_input)
print(postprocess(model_out))
