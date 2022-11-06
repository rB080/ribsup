import cv2
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader


def loader(imX_path, imY_path, mapX_path=None, mapY_path=None):
    imX = cv2.imread(imX_path)
    imY = cv2.imread(imY_path)
    if mapX_path is not None:
        mapX = cv2.imread(mapX_path, cv2.IMREAD_GRAYSCALE)
    if mapY_path is not None:
        mapY = cv2.imread(mapY_path, cv2.IMREAD_GRAYSCALE)

    imX = cv2.resize(imX, (256, 256), cv2.INTER_AREA)
    imY = cv2.resize(imY, (256, 256), cv2.INTER_AREA)

    imX = np.array(imX, np.float32).transpose(2, 0, 1) / 255.0
    imY = np.array(imY, np.float32).transpose(2, 0, 1) / 255.0
    if mapX_path is not None:
        mapX = np.array([mapX], np.float32) / 255.0
    if mapY_path is not None:
        mapY = np.array([mapY], np.float32) / 255.0
    if mapX_path is not None and mapY_path is not None:
        return imX, imY, mapX, mapY
    else:
        return imX, imY


def read_dataset(root_path, load_masks=True):
    imX = []
    imY = []

    imX_root = os.path.join(root_path, 'JSRT', 'JSRT')
    imY_root = os.path.join(root_path, 'BSE_JSRT', 'BSE_JSRT')

    for image_name in sorted(os.listdir(imY_root)):
        # .split('.')[0] + '.jpg')
        imX_path = os.path.join(imX_root, image_name)
        imY_path = os.path.join(imY_root, image_name)
        if os.path.exists(imX_path) and os.path.exists(imY_path):
            imX.append(imX_path)
            imY.append(imY_path)
    if load_masks:
        mapX = []
        mapY = []
        mx_root = os.path.join(root_path, 'mapX')
        my_root = os.path.join(root_path, 'mapY')
        for map_name in sorted(os.listdir(my_root)):
            mapX_path = os.path.join(mx_root, map_name)
            mapY_path = os.path.join(my_root, map_name)
            if os.path.exists(mapX_path) and os.path.exists(mapY_path):
                mapX.append(mapX_path)
                mapY.append(mapY_path)
        return imX, imY, mapX, mapY
    return imX, imY


class Translation_Dataset(Dataset):

    def __init__(self, root_path, split="train", get_maps=True):
        assert split in ["train", "test", "all"], "Invalid split!"
        self.root = root_path
        self.map_requirement = get_maps

        if get_maps:
            self.imX, self.imY, self.mapX, self.mapY = read_dataset(self.root)
            split_point = 9 * len(self.imX) // 10
            if split == "train":
                self.imX, self.imY, self.mapX, self.mapY = self.imX[:split_point], self.imY[
                    :split_point], self.mapX[:split_point], self.mapY[:split_point]
            elif split == "test":
                self.imX, self.imY, self.mapX, self.mapY = self.imX[split_point:], self.imY[
                    split_point:], self.mapX[split_point:], self.mapY[split_point:]
            # self.imX, self.imY, self.mapX, self.mapY = self.imX[0:1], self.imY[
            #    0:1], self.mapX[0:1], self.mapY[0:1]  # for cpu testing
            print('num imX = ', len(self.imX))
            print('num imY = ', len(self.imY))
            print('num maps of each kind = ', len(self.mapX), len(self.mapY))
        else:
            self.imX, self.imY = read_dataset(self.root, load_masks=False)
            split_point = 9 * len(self.imX) // 10
            if split == "train":
                self.imX, self.imY = self.imX[:split_point], self.imY[:split_point]
            elif split == "test":
                self.imX, self.imY = self.imX[split_point:], self.imY[split_point:]
            print('num imX = ', len(self.imX))
            print('num imY = ', len(self.imY))

    def __getitem__(self, index):
        if self.map_requirement:
            imX, imY, mapX, mapY = loader(
                self.imX[index], self.imY[index], self.mapX[index], self.mapY[index])
        else:
            imX, imY = loader(self.imX[index], self.imY[index])
        imX = torch.tensor(imX, dtype=torch.float32)
        imY = torch.tensor(imY, dtype=torch.float32)
        if self.map_requirement:
            mapX = torch.tensor(mapX, dtype=torch.float32)
            mapY = torch.tensor(mapY, dtype=torch.float32)
            pack = {"imX": imX, "imY": imY, "mapX": mapX,
                    "mapY": mapY, "name": self.imX[index]}
        else:
            pack = {"imX": imX, "imY": imY, "name": self.imX[index]}
        return pack

    def __len__(self):
        assert len(self.imX) == len(
            self.imY), 'The number of imX must be equal to imY'
        return len(self.imX)


def get_loader(root_path, split="train", get_maps=True, batch_size=1, shuffle=True):
    dataset = Translation_Dataset(root_path, split, get_maps)
    length = len(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, length, loader
