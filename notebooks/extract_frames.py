import torch
from torch import nn
from pretrainedmodels import bninception
from torchvision import transforms
from glob import glob
from PIL import Image
# import lmdb
from tqdm import tqdm
from os.path import basename
from argparse import ArgumentParser
import numpy as np

import os
import cv2
import shutil

import multiprocessing
from multiprocessing import Pool
def get_frames(path, name):
    try: 
        frames_dir_path = path.split(".")[0]

        if os.path.exists(frames_dir_path):
            cap = cv2.VideoCapture(path)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            files_count = len(os.listdir(frames_dir_path))
            cap.release()
            if length == files_count:
                print("Already Extracted:", name)
                return length, frames_dir_path
            else:
                shutil.rmtree(frames_dir_path)
        
        print("Started Extracting frames for: ", name)
        os.makedirs(frames_dir_path)

        vidcap = cv2.VideoCapture(path)
        success,image = vidcap.read()
        count = 0

        while success:
    #         print(image.shape)
            image = cv2.resize(image, (454,256))
    #         print(image.shape)
            cv2.imwrite(os.path.join(frames_dir_path, str(count) +".jpg"), image)
            success,image = vidcap.read()
            count+=1
            #todo: verify if count==frames_count

        print("Completed Extracting frames for: ", name)
        vidcap.release()
        return count, frames_dir_path
    except Exception as e:
        print(name, e)

            
def extract_frames(params):
    src_vids_path = params["path"]
    subjects = params["subjects"]
    
    for subject in subjects:
        spath = os.path.join(src_vids_path, subject)        
        viewpoints = os.listdir(spath)
        viewpoints = [viewpoint for viewpoint in viewpoints if ".avi" in viewpoint]
        for viewpoint in viewpoints:
            feat_filename = viewpoint.split("-")[0]
            imgs_count, imgs_path = get_frames(os.path.join(src_vids_path, subject, viewpoint), feat_filename)