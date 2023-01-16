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

def load(img_path=None, map_path=None):
    assert img_path is not None or map_path is not None, "Two empty parameters!"
    if img_path is not None:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256), cv2.INTER_AREA)
        img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0
    if map_path is not None:
        mask = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256), cv2.INTER_AREA)
        mask = np.array([mask], np.float32) / 255.0

    if img_path is not None and map_path is None: return img
    elif img_path is None and map_path is not None: return mask
    else: return img, mask

def read_dataset(root_path, load_masks=True, train=True, unpair=False):
    imX = []
    imY = []

    if train:
        imX_root = os.path.join(root_path, "Train", 'trainX')
        imY_root = os.path.join(root_path, "Train", 'trainY')
    else: 
        imX_root = os.path.join(root_path, "Test", 'testX')
        imY_root = os.path.join(root_path, "Test", 'testY')

    if not unpair:
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
            if train:
                mx_root = os.path.join(root_path, "Train", 'mapX')
                my_root = os.path.join(root_path, "Train", 'mapY')
            else:
                mx_root = os.path.join(root_path, "Test", 'mapX')
                my_root = os.path.join(root_path, "Test", 'mapY')
            
            for map_name in sorted(os.listdir(my_root)):
                mapX_path = os.path.join(mx_root, map_name)
                mapY_path = os.path.join(my_root, map_name)
                if os.path.exists(mapX_path) and os.path.exists(mapY_path):
                    mapX.append(mapX_path)
                    mapY.append(mapY_path)
            return imX, imY, mapX, mapY
        return imX, imY
    
    else: #Unpaired case
        for image_name in sorted(os.listdir(imY_root)): 
            imY_path = os.path.join(imY_root, image_name)
            imY.append(imY_path)
        for image_name in sorted(os.listdir(imX_root)): 
            imX_path = os.path.join(imX_root, image_name)
            imX.append(imX_path)
        if load_masks:
            mapX = []
            mapY = []
            if train:
                mx_root = os.path.join(root_path, "Train", 'mapX')
                my_root = os.path.join(root_path, "Train", 'mapY')
            else:
                mx_root = os.path.join(root_path, "Test", 'mapX')
                my_root = os.path.join(root_path, "Test", 'mapY')
            for map_name in sorted(os.listdir(my_root)):
                mapY_path = os.path.join(my_root, map_name)
                mapY.append(mapY_path)
            for map_name in sorted(os.listdir(mx_root)):
                mapX_path = os.path.join(mx_root, map_name)
                mapX.append(mapX_path)
            return imX, imY, mapX, mapY
        return imX, imY




class Translation_Dataset_Unpaired(Dataset):

    def __init__(self, root_path, split="train", get_maps=True):
        assert split in ["train", "test"], "Invalid split!"
        self.root = root_path
        self.map_requirement = get_maps
        self.split_point = 0
        if get_maps:
            self.I, self.M = [], []
            if split == "train": imX, imY, mapX, mapY = read_dataset(self.root, unpair=True)
            else: imX, imY, mapX, mapY = read_dataset(self.root, train=False, unpair=True)
            self.split_point = len(imX)
            print('num imX = ', len(imX))
            print('num imY = ', len(imY))
            print('num maps of each kind = ', len(mapX), len(mapY))
            self.I = imX + imY
            self.M = mapX + mapY
        
        else:
            self.I = []
            if split == "train": imX, imY = read_dataset(self.root, load_masks=False, unpair=True)
            else: imX, imY = read_dataset(self.root, load_masks=False, train=False, unpair=True)
            print('num imX = ', len(imX))
            print('num imY = ', len(imY))
            self.I = imX + imY
            self.split_point = len(imX)

    def __getitem__(self, index):
        if self.map_requirement:
            #print(len(self.imX), len(self.imY), len(self.mapX), len(self.mapY), index)
            img, mask = load(
                self.I[index], self.M[index])
        else:
            img = load(self.I[index])
        img = torch.tensor(img, dtype=torch.float32)
        
        if index < self.split_point: img_type = "X"
        else: img_type = "Y"

        if self.map_requirement:
            mask = torch.tensor(mask, dtype=torch.float32)
            pack = {"img": img, "map": mask,
                    "name": self.I[index], "type": img_type}
        else:
            pack = {"img": img, "name": self.I[index], "type": img_type}
        return pack

    def __len__(self):
        return len(self.I)




class Translation_Dataset(Dataset):

    def __init__(self, root_path, split="train", get_maps=True):
        assert split in ["train", "test"], "Invalid split!"
        self.root = root_path
        self.map_requirement = get_maps

        if get_maps:
            if split == "train": self.imX, self.imY, self.mapX, self.mapY = read_dataset(self.root)
            else: self.imX, self.imY, self.mapX, self.mapY = read_dataset(self.root, train=False)
            print('num imX = ', len(self.imX))
            print('num imY = ', len(self.imY))
            print('num maps of each kind = ', len(self.mapX), len(self.mapY))
        else:
            if split == "train": self.imX, self.imY = read_dataset(self.root, load_masks=False)
            else: self.imX, self.imY = read_dataset(self.root, load_masks=False, train=False)
            print('num imX = ', len(self.imX))
            print('num imY = ', len(self.imY))

    def __getitem__(self, index):
        if self.map_requirement:
            #print(len(self.imX), len(self.imY), len(self.mapX), len(self.mapY), index)
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


def get_loader(args, split="train", get_maps=True, shuffle=True):
    dataset = Translation_Dataset(args.data_root, split, get_maps)
    length = len(dataset)
    loader = DataLoader(dataset, batch_size=args.trans_batch, shuffle=shuffle, num_workers=args.num_workers)
    return dataset, length, loader


def get_unpaired_loader(args, split="train", get_maps=True, shuffle=False):
    dataset = Translation_Dataset_Unpaired(args.data_root, split, get_maps)
    length = len(dataset)
    loader = DataLoader(dataset, batch_size=args.trans_batch, shuffle=shuffle, num_workers=args.num_workers)
    return dataset, length, loader

