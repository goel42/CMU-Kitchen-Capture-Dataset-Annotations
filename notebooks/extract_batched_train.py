import os
import pickle
import numpy as np
from torch.utils.data import Dataset
import glob
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

class ImageLoader(Dataset):
	def __init__(self, frame_folder):
		self.data_folder = frame_folder + '/frames/' + "*.jpg"
		self.img_files = glob.glob(self.data_folder)
		self.total_files = len(self.img_files)
		self.scaler = transforms.Scale((224, 224))
		self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		self.to_tensor = transforms.ToTensor()

	def __len__(self):
		return self.total_files

	def __getitem__(self, idx):
		file_name = self.img_files[idx]

		img_index = file_name.split('/')[-1]
		img_index = int(img_index.split('.')[0])

		img = cv2.imread(file_name)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		img = Image.fromarray(img)
		img = self.scaler(img)

		img = self.to_tensor(img)
		img = self.normalize(img)
		tensor = Variable(img)
		return tensor, img_index


def hook(module, input, output):
	N,C,H,W = output.shape
	output = output.squeeze()
	features.append(output.cpu().detach().numpy())

model_name = 'resnet101'
layer_name = 'avgpool'

get_model = getattr(torchvision.models, model_name)
model = get_model(pretrained=True)
model = model.cuda()
handle = model._modules.get(layer_name).register_forward_hook(hook)

batch_s = 32

if __name__ == "__main__":
	recipe_path = '/mnt/data/tasty_data/ALL_RECIPES/'
	dest_path = '/mnt/data/tasty_data/resnet_feat/'
	all_recipes = [all_recipes.rstrip('\n')    for all_recipes  in open( '/mnt/data/tasty_data/TEST_SET.txt')]

	for kki in range( len(all_recipes) ):
		t1 = datetime.now()

		if(kki % 100 == 0):
			print (kki)
		dataset = ImageLoader(recipe_path + all_recipes[kki])
		total_files = len(dataset)

		resnet_vector = np.zeros((total_files, 2048))
		trainloader = DataLoader(dataset, batch_size=batch_s, shuffle=False, num_workers=4, drop_last=False)
		
		for i, data in enumerate(trainloader):
			inputs, img_index = data
			inputs = inputs.cuda()

			features = []
			outputs = model(inputs)
			features = np.concatenate(features)

			for j in range(inputs.shape[0]):
				resnet_vector[img_index[j]] = features[j]

		dest_folder = dest_path + all_recipes[kki]
		try:
			os.makedirs(dest_folder)
		except:
			print ("No")
		dest_np = dest_folder + "/" + "resnet101.npy"
		np.save(dest_np, resnet_vector)

		t2 = datetime.now()
		delta = t2 - t1
		print (delta.total_seconds())

