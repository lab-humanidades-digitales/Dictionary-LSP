from __future__ import print_function

import argparse
import os
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as f
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# 27 points
selected_joints =  np.concatenate(([0,5,6,7,8,9,10],
                                   [91,95,96,99,100,103,104,107,108,111],
                                   [112,116,117,120,121,124,125,128,129,132]), axis=0)
num_joints = 27

max_body_true = 1
max_frame = 150
num_channels = 2#3

index_mirror = np.concatenate([
                [1,3,2,5,4,7,6,9,8,11,10,13,12,15,14,17,16],
                [21,22,23,18,19,20],
                np.arange(40,23,-1), np.arange(50,40,-1),
                np.arange(51,55), np.arange(59,54,-1),
                [69,68,67,66,71,70], [63,62,61,60,65,64],
                np.arange(78,71,-1), np.arange(83,78,-1),
                [88,87,86,85,84,91,90,89],
                np.arange(113,134), np.arange(92,113)]) - 1

assert(index_mirror.shape[0] == 133)

multi_scales = [512,640]


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Decoupling Graph Convolution Network with DropGraph Module')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument('-Experiment_name', default='')
    parser.add_argument(
        '--config',
        default='./config/test_joint.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='test', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default='./checkpoint/sign_joint.pt',
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=1,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--keep_rate',
        type=float,
        default=0.9,
        help='keep probability for drop')
    parser.add_argument(
        '--groups',
        type=int,
        default=8,
        help='decouple groups')
    parser.add_argument('--only_train_part', default=True)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    return parser

def norm_numpy_totensor(img):
    img = img.astype(np.float32) / 255.0
    for i in range(3):
        img[:, :, :, i] = (img[:, :, :, i] - mean[i]) / std[i]
    return torch.from_numpy(img).permute(0, 3, 1, 2)

def stack_flip(img):
    img_flip = cv2.flip(img, 1)
    return np.stack([img, img_flip], axis=0)

def merge_hm(hms_list):
    assert isinstance(hms_list, list)
    for hms in hms_list:
        hms[1,:,:,:] = torch.flip(hms[1,index_mirror,:,:], [2])
    
    hm = torch.cat(hms_list, dim=0)
    # print(hm.size(0))
    hm = torch.mean(hms, dim=0)
    return hm

def preproccess():

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

            pred = pred[selected_joints,:2]

            assert pred.shape == (27, 2)

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
            fp[pos,:,:,:,0] = skel[:max_frame,:,:]

        return fp


def init_seed(_):
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    #torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):

        arg.model_saved_name = "./save_models/" + arg.Experiment_name
        arg.work_dir = "./work_dir/" + arg.Experiment_name
        self.arg = arg

        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        #self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_tmp_acc = 0

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device =  self.arg.device[0] if type(
             self.arg.device) is list else self.arg.device
        self.output_device =  output_device
        Model = import_class(self.arg.model)
        self.model = Model(**self.arg.model_args).cpu()
        # print(self.model)

        self.loss = nn.CrossEntropyLoss().cpu()
        # self.loss = LabelSmoothingCrossEntropy().cuda(output_device)

        if self.arg.weights:
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cpu()] for k, v in weights.items()])

            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log('Sucessfully Remove Weights: {}.'.format(w))
                else:
                    self.print_log('Can Not Remove Weights: {}.'.format(w))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)

    def load_optimizer(self):

        if self.arg.optimizer == 'SGD':

            params_dict = dict(self.model.named_parameters())
            params = []

            for key, value in params_dict.items():
                decay_mult = 0.0 if 'bias' in key else 1.0

                lr_mult = 1.0
                weight_decay = 1e-4

                params += [{'params': value, 'lr': self.arg.base_lr, 'lr_mult': lr_mult,
                            'decay_mult': decay_mult, 'weight_decay': weight_decay}]

            self.optimizer = optim.SGD(
                params,
                momentum=0.9,
                nesterov=self.arg.nesterov)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)

        else:
            raise ValueError()

        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1,
                                              patience=10, verbose=True,
                                              threshold=1e-4, threshold_mode='rel',
                                              cooldown=0)


    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                    0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()


    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)


    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)


    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time


    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def eval(self, epoch, x, save_score=False, loader_name=['test'], isTest=False):

        #if isTest:
        submission = dict()
        trueLabels = dict()
        
        x = torch.Tensor(x)

        self.model.eval()
        with torch.no_grad():
            for ln in loader_name:

                data = Variable(
                    x.float().cpu(),
                    requires_grad=False)

                data = data.transpose(3,1).transpose(2,3)

                with torch.no_grad():
                    output = self.model(data)

                if isinstance(output, tuple):
                    output, l1 = output
                    l1 = l1.mean()
                else:
                    l1 = 0
                print(output.data)
                #_, predict_label = torch.max(output.data, 1)
                _, predict_label = torch.topk(output.data, 5)
                
                predict_label = predict_label[0]

        return predict_label.numpy() 


    def start(self, data):

        if self.arg.weights is None:
            raise ValueError('Please appoint --weights.')

        self.arg.print_log = False
        self.print_log('Model:   {}.'.format(self.arg.model))
        self.print_log('Weights: {}.'.format(self.arg.weights))
        return self.eval(epoch=self.arg.start_epoch,x = data, save_score=self.arg.save_score,
                    loader_name=['test'], isTest=True)


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


#if __name__ == '__main__':
data = preproccess()

parser = get_parser()

meaning = pd.read_json("meaning.json")
meaning = dict(meaning[0])
meaning = dict((v,k) for k,v in meaning.items())

# load arg form config file
p = parser.parse_args()
if p.config is not None:
    with open(p.config, 'r') as f:
        #default_arg = yaml.load(f)
        default_arg = yaml.safe_load(f)
    key = vars(p).keys()
    for k in default_arg.keys():
        if k not in key:
            print('WRONG ARG: {}'.format(k))
            assert (k in key)
    parser.set_defaults(**default_arg)
arg = parser.parse_args()

init_seed(0)
processor = Processor(arg)
result = processor.start(data)

result = [meaning[val] for val in result]

print()
print(result)