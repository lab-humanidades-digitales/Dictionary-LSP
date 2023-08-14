import gc
import ast
import tqdm
import time
import h5py
import glob
import torch
import pandas as pd
import numpy as np
from collections import Counter
import torch.utils.data as torch_data
from torch.utils.data import Dataset
import logging

def get_data_from_h5(path):
    hf = h5py.File(path, 'r')
    return hf

def get_dataset_from_hdf5(path,keypoints_model,landmarks_ref,keypoints_number,threshold_frecuency_labels=10,list_labels_banned=[],dict_labels_dataset=None,
                         inv_dict_labels_dataset=None):
    print('path                       :',path)
    print('keypoints_model            :',keypoints_model)
    print('landmarks_ref              :',landmarks_ref)
    print('threshold_frecuency_labels :',threshold_frecuency_labels)
    print('list_labels_banned         :',list_labels_banned)
    
    index_array_column = None #'mp_indexInArray', 'wp_indexInArray','op_indexInArray'

    print('Use keypoint model : ',keypoints_model) 
    if keypoints_model == 'openpose':
        index_array_column  = 'op_indexInArray'
    if keypoints_model == 'mediapipe':
        index_array_column  = 'mp_indexInArray'
    if keypoints_model == 'wholepose':
        index_array_column  = 'wp_indexInArray'
    print('use column for index keypoint :',index_array_column)

    assert not index_array_column is None

    df_keypoints = pd.read_csv(landmarks_ref, skiprows=1)
    if keypoints_number == 29:
        df_keypoints = df_keypoints[(df_keypoints['Selected 29']=='x' )& (df_keypoints['Key']!='wrist')]
    else:
        df_keypoints = df_keypoints[(df_keypoints['Selected73']=='x' )& (df_keypoints['Key']!='wrist')]

    logging.info(" using keypoints_number: "+str(keypoints_number))

    idx_keypoints = sorted(df_keypoints[index_array_column].astype(int).values)
    name_keypoints = df_keypoints['Key'].values
    section_keypoints = (df_keypoints['Section']+'_'+df_keypoints['Key']).values
    print('section_keypoints : ',len(section_keypoints),' -- uniques: ',len(set(section_keypoints)))
    print('name_keypoints    : ',len(name_keypoints),' -- uniques: ',len(set(name_keypoints)))
    print('idx_keypoints     : ',len(idx_keypoints),' -- uniques: ',len(set(idx_keypoints)))
    print('')
    print('section_keypoints used:')
    print(section_keypoints)


    print('Reading dataset .. ')
    data = get_data_from_h5(path)
    #torch.Size([5, 71, 2])
    print('Total size dataset : ',len(data.keys()))
    video_dataset  = []
    labels_dataset = []

    time.sleep(2)
    for index in tqdm.tqdm(list(data.keys())):

        data_video = np.array(data[index]['data'])
        data_label = np.array(data[index]['label']).item().decode('utf-8')

        n_frames,n_axis,n_keypoints = data_video.shape

        data_video = np.transpose(data_video, (0,2,1)) #transpose to n_frames, n_keypoints, n_axis 
        if index=='0':
            print('original size video : ',data_video.shape,'-- label : ',data_label)
            print('filtering by keypoints idx .. ')
        data_video = data_video[:,idx_keypoints,:]
        if index=='0':
            print('filtered size video : ',data_video.shape,'-- label : ',data_label)

        video_dataset.append(data_video)
        labels_dataset.append(data_label)
    del data
    gc.collect()
    

    
    print('frecuency labels filtering ...')
    hist_labels = dict(Counter(labels_dataset))
    print('hist counter')
    print(hist_labels)
    
    labels_high_frecuency = []
    for name in hist_labels.keys():
        if hist_labels[name] >= threshold_frecuency_labels and not name in list_labels_banned:
            labels_high_frecuency.append(name)
    labels_high_frecuency = sorted(labels_high_frecuency)
    len(labels_high_frecuency)
    print('total unique labels : ',len(set(labels_dataset)))
    
    filtros = [label in labels_high_frecuency for label in labels_dataset]
    
    print('before filter size video_dataset   :',len(video_dataset))
    print('before filter size labels_dataset  :',len(labels_dataset))
    #print('before filter size encoded_dataset :',len(encoded_dataset))
    video_dataset   = np.array(video_dataset)[filtros]
    labels_dataset  = np.array(labels_dataset)[filtros]
    #encoded_dataset = np.array(encoded_dataset)[filtros]
    print('after  filter size video_dataset   :',len(video_dataset))
    print('after  filter size labels_dataset  :',len(labels_dataset))
    #print('after  filter size encoded_dataset :',len(encoded_dataset))
    print('frecuency labels completed!')
    print('label encoding ...')
    
    if dict_labels_dataset is None:
        dict_labels_dataset = {}
        inv_dict_labels_dataset = {}

        for index,label in enumerate(sorted(set(labels_dataset))):
            dict_labels_dataset[label] = index
            inv_dict_labels_dataset[index] = label
    
    print('sorted(set(labels_dataset))  : ',sorted(set(labels_dataset)))
    print('dict_labels_dataset      :',dict_labels_dataset)
    print('inv_dict_labels_dataset  :',inv_dict_labels_dataset)
    encoded_dataset = [dict_labels_dataset[label] for label in labels_dataset]
    print('encoded_dataset:',len(encoded_dataset))

    print('label encoding completed!')

    print('total unique labels : ',len(set(labels_dataset)))
    print('Reading dataset completed!')

    return video_dataset,labels_dataset,encoded_dataset,dict_labels_dataset,inv_dict_labels_dataset

class LSP_Dataset(Dataset):
    """Advanced object representation of the HPOES dataset for loading hand joints landmarks utilizing the Torch's
    built-in Dataset properties"""

    data: [np.ndarray]  # type: ignore
    labels: [np.ndarray]  # type: ignore

    def __init__(self, dataset_filename: str,keypoints_model:str, num_labels=5, transform=None, augmentations=False,
                 augmentations_prob=0.5, normalize=False,landmarks_ref= 'Data/Mapeo landmarks librerias - Hoja 1_2.csv',
                dict_labels_dataset=None,inv_dict_labels_dataset=None,keypoints_number = 29):
        """
        Initiates the HPOESDataset with the pre-loaded data from the h5 file.

        :param dataset_filename: Path to the h5 file
        :param transform: Any data transformation to be applied (default: None)
        """
        print("*"*20)
        print("*"*20)
        print("*"*20)
        print('Use keypoint model : ',keypoints_model) 
        logging.info('Use keypoint model : '+str(keypoints_model))

        if  'AEC' in  dataset_filename:
            self.list_labels_banned = ["ya", "qué?", "qué", "bien", "dos", "ahí", "luego", "yo", "él", "tú","???","NNN"]

        if  'PUCP' in  dataset_filename:
            self.list_labels_banned = ["ya", "qué?", "qué", "bien", "dos", "ahí", "luego", "yo", "él", "tú","???","NNN"]
            self.list_labels_banned += ["sí","ella","uno","ese","ah","dijo","llamar"]

        if  'WLASL' in  dataset_filename:
            self.list_labels_banned = ['apple','computer','fish','kiss','later','no','orange','pizza','purple','secretary','shirt','sunday','take','water','yellow']

        print('self.list_labels_banned',self.list_labels_banned)
        logging.info('self.list_labels_banned '+str(self.list_labels_banned))

        video_dataset,labels_dataset,encoded_dataset,dict_labels_dataset,inv_dict_labels_dataset = get_dataset_from_hdf5(path=dataset_filename,
                                                                                        keypoints_model=keypoints_model,
                                                                                        landmarks_ref=landmarks_ref,
                                                                                         keypoints_number = keypoints_number,
                                                                                         threshold_frecuency_labels =0,
                                                                                         list_labels_banned =self.list_labels_banned,
                                                                                        dict_labels_dataset=dict_labels_dataset,
                                                                                         inv_dict_labels_dataset=inv_dict_labels_dataset)

        self.data = video_dataset
        self.labels = encoded_dataset
        #self.targets = list(encoded_dataset)
        self.text_labels = list(labels_dataset)
        self.num_labels = num_labels
        self.transform = transform
        self.dict_labels_dataset = dict_labels_dataset
        self.inv_dict_labels_dataset = inv_dict_labels_dataset
        

        self.augmentations = augmentations
        self.augmentations_prob = augmentations_prob
        self.normalize = normalize

    def __getitem__(self, idx):
        """
        Allocates, potentially transforms and returns the item at the desired index.

        :param idx: Index of the item
        :return: Tuple containing both the depth map and the label
        """
        depth_map = torch.from_numpy(np.copy(self.data[idx]))
        label = torch.Tensor([self.labels[idx]])
        depth_map = depth_map - 0.5
        if self.transform:
            depth_map = self.transform(depth_map)
        return depth_map, label

    def __len__(self):
        return len(self.labels)
