import cv2
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader


def loader(img_path, map_path):
    img = cv2.imread(img_path)
    map = cv2.imread(map_path)
    img = cv2.resize(img, (256, 256), cv2.INTER_AREA)
    map = cv2.resize(map, (256, 256), cv2.INTER_AREA)

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0
    map = np.array(map, np.float32).transpose(2, 0, 1) / 255.0
    return img, map


def read_dataset(root_path):
    images = []
    maps = []

    img_root = os.path.join(root_path, 'JSRT')
    map_root = os.path.join(root_path, 'BSE_JSRT')

    for image_name in sorted(os.listdir(map_root)):
        img_path = os.path.join(img_root, image_name)
        map_path = os.path.join(map_root, image_name)
        if os.path.exists(img_path) and os.path.exists(map_path):
            images.append(img_path)
            maps.append(map_path)
    return images, maps


class Segmentation_Dataset(Dataset):

    def __init__(self, root_path):
        self.root = root_path
        self.images, self.maps = read_dataset(self.root)
        print('num img = ', len(self.img))
        print('num map = ', len(self.map))
        print('num maps of each kind = ', len(self.mapX), len(self.mapY))

    def __getitem__(self, index):
        img, map = loader(self.img[index], self.map[index])
        img = torch.tensor(img, dtype=torch.float32)
        map = torch.tensor(map, dtype=torch.float32)
        pack = {"img": img, "map": map}
        return pack

    def __len__(self):
        assert len(self.img) == len(
            self.map), 'The number of img must be equal to map'
        return len(self.img)


def get_loader(root_path, batch_size=1, shuffle=True):
    dataset = Segmentation_Dataset(root_path)
    length = len(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, length, loader
