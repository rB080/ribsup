import cv2
import numpy as np
import torch
import os
from torch.utils.data import Dataset


def loader(img_path, mask_path, mapX_path, mapY_path, phase):
    # print('image: ', img_path)
    # print('mask: ', mask_path)

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)  # , cv2.IMREAD_GRAYSCALE
    mapX = cv2.imread(mapX_path, cv2.IMREAD_GRAYSCALE)
    mapY = cv2.imread(mapY_path, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (256, 256), cv2.INTER_AREA)
    mask = cv2.resize(mask, (256, 256), cv2.INTER_AREA)

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0
    # print(mask.shape)
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mapX = np.array([mapX], np.float32) / 255.0
    mapY = np.array([mapY], np.float32) / 255.0
    return img, mask, mapX, mapY


def read_dataset(root_path, mode):
    images = []
    masks = []
    mapX = []
    mapY = []
    image_root = os.path.join(root_path, 'JSRT')
    gt_root = os.path.join(root_path, 'BSE_JSRT')
    mx_root = os.path.join(root_path, 'JSRT_maps')
    my_root = os.path.join(root_path, 'BSE_JSRT_maps')

    print(len(os.listdir(image_root)), len(os.listdir(gt_root)))
    for image_name in sorted(os.listdir(gt_root)):
        image_path = os.path.join(image_root, image_name)  # .split('.')[0] + '.jpg')
        label_path = os.path.join(gt_root, image_name)
        # assert os.path.exists(image_path), 'X image path: '+image_path+' is invalid...'
        # assert os.path.exists(label_path), 'Y image path: '+label_path+' is invalid...'
        if os.path.exists(image_path) and os.path.exists(label_path):
            images.append(image_path)
            masks.append(label_path)

    for map_name in sorted(os.listdir(my_root)):
        mapX_path = os.path.join(mx_root, map_name)
        mapY_path = os.path.join(my_root, map_name)
        # assert os.path.exists(mapX_path), 'X map path: '+mapX_path+' is invalid...'
        # assert os.path.exists(mapY_path), 'Y map path: '+mapY_path+' is invalid...'
        if os.path.exists(mapX_path) and os.path.exists(mapY_path):
            mapX.append(mapX_path)
            mapY.append(mapY_path)

    return images, masks, mapX, mapY


class Eye_Dataset(Dataset):

    def __init__(self, root_path, phase):
        self.root = root_path
        self.phase = phase
        self.images, self.labels, self.mapX, self.mapY = read_dataset(self.root, self.phase)
        print('num images = ', len(self.images))
        print('num labels = ', len(self.labels))
        print('num maps of each kind = ', len(self.mapX), len(self.mapY))

    def __getitem__(self, index):
        img, mask, mapX, mapY = loader(self.images[index], self.labels[index], self.mapX[index], self.mapY[index],
                                       self.phase)
        img = torch.tensor(img, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        mapX = torch.tensor(mapX, dtype=torch.float32)
        mapY = torch.tensor(mapY, dtype=torch.float32)
        return img, mask, mapX, mapY

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)

