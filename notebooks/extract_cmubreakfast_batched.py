import os
import pickle
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import pandas as pd
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datetime import datetime

from pretrainedmodels import bninception


class ImageLoader(Dataset):
    def __init__(self, frames_dir):
        self.frames_dir = frames_dir
        self.frames = glob(os.path.join(self.frames_dir, "*.jpg"))
        self.frames_count = len(self.frames)
        
        self.transform = transforms.Compose([
            transforms.Resize([256, 454]),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[[2,1,0],...]*255), #to BGR
            transforms.Normalize(mean=[104, 117, 128],std=[1, 1, 1]),
        ])
    
    def __getitem__(self,idx):
        file_name = str(idx)+".jpg"
        file_path = os.path.join(self.frames_dir, file_name)
        
        frame = cv2.imread(file_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame = Image.fromarray(frame)
        frame = self.transform(frame) #to device cuda?#DOUBT?
        
        return frame, idx
        
    
    def __len__(self):
        return self.frames_count

torch.cuda.set_device(5) #brownie
# torch.cuda.set_device(2) #for sandwich
# torch.cuda.set_device(4) #for eggs
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = bninception(pretrained=None)
state_dict = torch.load('rulstm/FEATEXT/models/TSN-rgb.pth.tar', map_location='cpu')['state_dict']
state_dict = {k.replace('module.base_model.','') : v for k,v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)


model.last_linear = nn.Identity()
model.global_pool = nn.AdaptiveAvgPool2d(1)

model.to(device)
model.eval()


batch_s = 32
activity = "Brownie"
src_vids_path = os.path.join("/home/anujraj/cmu-kitchen-capture/raw/", activity)
dest_feats_path = os.path.join("/mnt/data/ar-datasets/cmu-kitchen/features2", activity)
if not os.path.exists(dest_feats_path):
    os.makedirs(dest_feats_path)

subjects = os.listdir(src_vids_path)
subjects = [subject for subject in subjects if "Video" in subject]
subjects = sorted(subjects)


for subject in subjects:
        spath = os.path.join(src_vids_path, subject)
        
        viewpoints = os.listdir(spath)
        viewpoints = [viewpoint for viewpoint in viewpoints if ".avi" in viewpoint]
        for viewpoint in viewpoints:
            feat_filename = viewpoint.split("-")[0]
            if feat_filename+".npy" in os.listdir(dest_feats_path):
                print("Already extracted: ", feat_filename)
                continue
               
            print("Started extracting feats for: ", feat_filename)
            feat_dim = 1024
            
            dataset = ImageLoader(os.path.join(src_vids_path, subject, viewpoint.split(".")[0]))
            total_frames = len(dataset)
            
            feats = np.zeros((total_frames, 1024))
            trainloader = DataLoader(dataset, batch_size=batch_s, shuffle=False, num_workers=16, drop_last=False)
            
            for i, data in enumerate(trainloader):
                frames, frames_indices = data
                frames = frames.to(device)
                
                curr_feats = model(frames)
                
                for j in range(frames.shape[0]):
                    feats[frames_indices[j]] = curr_feats[j].cpu().detach().numpy()
            
            np.save(os.path.join(dest_feats_path, feat_filename+".npy"), feats)
            print("Completed: ", feat_filename)
