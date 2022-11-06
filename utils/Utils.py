import torch
import cv2
import numpy as np

def convert(A, Ttnp = True):
    if Ttnp == True: return A.detach().cpu().numpy()
    else: return torch.tensor(A, device='cuda')

def disp_Tensor(map, scale=1, Tensor=True):
    if Tensor==True: map = convert(map)
    map = map[0,:,:,:].transpose(1,2,0)
    cv2.imshow('Image: ', map*255.0)

def inject_Noise(T, noise=0.5):
    T = convert(T)
    T = T + noise*np.random.rand(T.shape[0], T.shape[1], T.shape[2], T.shape[3])
    T = convert(T, False)
    return T

