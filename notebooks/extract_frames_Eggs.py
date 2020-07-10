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
from extract_frames import extract_frames
 
activity = "Eggs"
src_vids_path = os.path.join("/home/anujraj/cmu-kitchen-capture/raw/", activity )
# dest_feats_path = os.path.join("/home/anujraj/cmu-kitchen-capture/features")
# os.mkdir(dest_feats_path)
if __name__ == "__main__":
    
    subjects = os.listdir(src_vids_path)
    subjects = sorted([subject for subject in subjects if "Video" in subject])

    p = Pool(7)
    params = [{"path": src_vids_path, "subjects":subjects[i*5:(i*5)+5]} for i in range(7)]
    p.map(extract_frames, params)