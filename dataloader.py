import os.path as osp
import copy
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from PIL import ImageFile
import torchvision.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True

class dataset(nn.Module):
	def __init__(self, mode):
		self.mode = mode

		self.num_parts = 7
		self.pasta_names = ['verb', 'foot', 'leg', 'hip', 'hand', 'arm', 'head']
		self.anno_keys =   ['gt_verbs', 'gt_pasta_foot', 'gt_pasta_leg', 'gt_pasta_hip', 'gt_pasta_hand', 'gt_pasta_arm', 'gt_pasta_head']
		self.num_pastas =  [157, 16, 15, 6, 34, 8, 14]

		if mode == 'train':
			with open('data/hake/annotations/hake_train_img.json') as file:
				js = json.load(file)
			self.anno = js['annotations']
			self.tran = T.Compose([
            	T.Resize((224, 224)),
            	T.RandomHorizontalFlip(),
            	T.ToTensor(), 
				T.Normalize([0.482, 0.458, 0.408], [0.269, 0.261, 0.276])
        	])
		else:
			with open('data/hake/annotations/hake_val_img.json') as file:
				js = json.load(file)
			self.anno = js['annotations']
			self.tran = T.Compose([
            	T.Resize((224, 224)),
            	T.ToTensor(), 
				T.Normalize([0.482, 0.458, 0.408], [0.269, 0.261, 0.276])
        	])

	def __len__(self):
		return len(self.anno)

	def __getitem__(self, x):
		now_anno = copy.deepcopy(self.anno[x])
		image_path = osp.join('data/hake', now_anno['img_path'])

		image = Image.open(image_path).convert('RGB')
		image = self.tran(image)

		gt = {}

		for idx in range(self.num_parts):
			name = self.pasta_names[idx]
			key  = self.anno_keys[idx]
			num  = self.num_pastas[idx]

			gt_now = np.zeros(num, dtype=np.float32)
			gt_now[now_anno[key]] = 1
			gt[name] = gt_now
		
		for key in gt:
			gt[key] = torch.from_numpy(gt[key])

		return image, gt
