import ast
import torch

import pandas as pd
import torch.utils.data as torch_data

import numpy as np

#from augmentations import *
#from normalization.body_normalization import BODY_IDENTIFIERS
#from normalization.hand_normalization import HAND_IDENTIFIERS

BODY_IDENTIFIERS = [

'k0',
 'k16',
 'k15',
 'k5',
 'k2',
 'k6',
 'k3',
 'k7',
 'k4',
 'k75',
 'k46',
 'k73',
 'k42',
 'k81',
 'k44',
 'k61',
 'k28',
 'k64',
 'k32',
 'k30',
 'k77',
 'k47',
 'k79',
 'k51',
 'k83',
 'k49',
 'k70',
 'k38',
 'k67',
 'k34',
 'k36',
 'k96',
 'k97',
 'k98',
 'k99',
 'k100',
 'k101',
 'k102',
 'k103',
 'k104',
 'k105',
 'k106',
 'k107',
 'k108',
 'k109',
 'k110',
 'k111',
 'k112',
 'k113',
 'k114',
 'k115',
 'k117',
 'k118',
 'k119',
 'k120',
 'k121',
 'k122',

]

HAND_IDENTIFIERS = [

 'k123',
 'k124',
 'k125',
 'k126',
 'k127',
 'k128',
 'k129',
 'k130',
 'k131',
 'k132',
 'k133',
 'k134',
 'k135',
 'k136'

]


import torch


class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



#from normalization.body_normalization import normalize_single_dict as normalize_single_body_dict
#from normalization.hand_normalization import normalize_single_dict as normalize_single_hand_dict

HAND_IDENTIFIERS = [id for id in HAND_IDENTIFIERS]# + [id for id in HAND_IDENTIFIERS]


def load_dataset(file_location: str):

    # Load the datset csv file
    df = pd.read_csv(file_location, encoding="utf-8")

    # TO BE DELETED
    df.columns = [item.replace("_left_", "_0_").replace("_right_", "_1_") for item in list(df.columns)]
    
    #if "neck_X" not in df.columns:
    #    df["neck_X"] = [0 for _ in range(df.shape[0])]
    #    df["neck_Y"] = [0 for _ in range(df.shape[0])]

    # TEMP
    labels = df["labels"].to_list()

    labels = [label + 1 for label in df["labels"].to_list()]
    print('set labels')
    print(sorted(set(labels)))
    data = []
    print('BODY_IDENTIFIERS',BODY_IDENTIFIERS)
    print('HAND_IDENTIFIERS',HAND_IDENTIFIERS)
    #print('len(ast.literal_eval(row["k0_X"]))',len(ast.literal_eval(row["k0_X"])))
    for row_index, row in df.iterrows():
        
        current_row = np.empty(shape=(len(ast.literal_eval(row["k0_X"])), len(BODY_IDENTIFIERS + HAND_IDENTIFIERS), 2))
        for index, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
            #print(identifier)
            current_row[:, index, 0] = ast.literal_eval(row[identifier + "_X"])
            current_row[:, index, 1] = ast.literal_eval(row[identifier + "_Y"])

        data.append(current_row)

    return data, labels


def tensor_to_dictionary(landmarks_tensor: torch.Tensor) -> dict:

    data_array = landmarks_tensor.numpy()
    output = {}

    for landmark_index, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
        output[identifier] = data_array[:, landmark_index]

    return output


def dictionary_to_tensor(landmarks_dict: dict) -> torch.Tensor:

    output = np.empty(shape=(len(landmarks_dict["k0"]), len(BODY_IDENTIFIERS + HAND_IDENTIFIERS), 2))

    for landmark_index, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
        output[:, landmark_index, 0] = [frame[0] for frame in landmarks_dict[identifier]]
        output[:, landmark_index, 1] = [frame[1] for frame in landmarks_dict[identifier]]

    return torch.from_numpy(output)


class CzechSLRDataset(torch_data.Dataset):
    """Advanced object representation of the HPOES dataset for loading hand joints landmarks utilizing the Torch's
    built-in Dataset properties"""

    data: [np.ndarray]  # type: ignore
    labels: [np.ndarray]  # type: ignore

    def __init__(self, dataset_filename: str, num_labels=5, transform=None, augmentations=False,
                 augmentations_prob=0.5, normalize=False):
        """
        Initiates the HPOESDataset with the pre-loaded data from the h5 file.

        :param dataset_filename: Path to the h5 file
        :param transform: Any data transformation to be applied (default: None)
        """

        loaded_data = load_dataset(dataset_filename)
        data, labels = loaded_data[0], loaded_data[1]

        self.data = data
        self.labels = labels
        self.targets = list(labels)
        self.num_labels = num_labels
        self.transform = transform

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
        label = torch.Tensor([self.labels[idx] - 1])

        #label = torch.nn.functional.one_hot(label, num_classes=64)


        depth_map = tensor_to_dictionary(depth_map)

        # Apply potential augmentations
        '''
        if self.augmentations and random.random() < self.augmentations_prob:

            selected_aug = randrange(4)

            if selected_aug == 0:
                depth_map = augment_rotate(depth_map, (-13, 13))

            if selected_aug == 1:
                depth_map = augment_shear(depth_map, "perspective", (0, 0.1))

            if selected_aug == 2:
                depth_map = augment_shear(depth_map, "squeeze", (0, 0.15))

            if selected_aug == 3:
                depth_map = augment_arm_joint_rotate(depth_map, 0.3, (-4, 4))

        if self.normalize:
            depth_map = normalize_single_body_dict(depth_map)
            depth_map = normalize_single_hand_dict(depth_map)
        '''
        depth_map = dictionary_to_tensor(depth_map)

        # Move the landmark position interval to improve performance
        depth_map = depth_map - 0.5

        if self.transform:
            depth_map = self.transform(depth_map)

        return depth_map, label

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    pass
