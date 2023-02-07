import argparse

import os
import torch
import json
import numpy as np
from models import clip_a2v
from dataloader import dataset
from torch.utils.data import DataLoader
import util.misc as utils

def get_args_parser():
	parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
	parser.add_argument('--device', default='cuda',type=str)
	parser.add_argument('--pretrained', default='', type=str)

	parser.add_argument('--train', default=True, type=bool)
	parser.add_argument('--epochs', default=5, type=int)
	parser.add_argument('--batch_size', default=64, type=int)
	parser.add_argument('--lr', default=1e-3, type=float)
	parser.add_argument('--single_part', default='', type=str)

	parser.add_argument('--eval', default=False, action='store_true')
	return parser

def train(epoch, model, optim, lr_scheduler, data_loader_train, device, total):
	model.train()
	it = 0
	for image, gt in data_loader_train:
		image = image.to(device)
		s_parts, s_verb = model(image)
		loss_dict = model.get_loss(s_parts, s_verb, gt)
		losses = sum(loss_dict[k] for k in loss_dict.keys() if 'loss' in k)
		loss_dict_reduced = utils.reduce_dict(loss_dict)
		loss_dict_reduced = {k: v.item() for k, v in loss_dict_reduced.items()}

		optim.zero_grad()
		losses.backward()
		optim.step()
		lr_scheduler.step()

		it = it + 1
		if it % 50 == 0:
			print("Epoch [{}]: [{}/{}]".format(epoch, it, total), end=" ")
			for key in loss_dict.keys():
				print(key, end=": ")
				print(loss_dict[key], end=" ")
			print("")

@torch.no_grad()
def evaluate(model, data_loader_eval, device, total):
	model.eval()
	print("Evaluate:")

	it = 0
	results_pasta = []
	results_verb = []
	gt_labels = {
		f'verb': [],
		f'foot': [],
		f'leg':	 [],
		f'hip':  [],
		f'hand': [],
		f'arm':  [],
		f'head': []
	}
	for image, gt in data_loader_eval:
		image = image.to(device)
		p_pasta, p_verb = model(image)

		results_pasta.append(p_pasta[0].cpu().numpy())
		results_verb.append(p_verb[0].cpu().numpy())
		for key in gt_labels.keys():
			tmp = gt[key][0].cpu().numpy()
			gt_labels[key].append(tmp)

		it = it + 1
		if it % 1000 == 0:
			print("[{}/{}]".format(it, total))

	results_pasta = np.array(results_pasta)
	results_verb = np.array(results_verb)
	for key in gt_labels.keys():
		gt_labels[key] = np.array(gt_labels[key])

	score = model.get_map(results_pasta, results_verb, gt_labels)
	print(score)
	return score

def save_model(epoch, model, optim, lr_scheduler):
	path = os.path.join("results", "epoch" + str(epoch) + ".pth")
	if not os.path.exists('results'):
		os.mkdir(r'results')
	file = open(path, 'w')
		
	torch.save({'epoch': epoch,
				'model': model.state_dict(),
				'optim': optim.state_dict(),
				'lr_scheduler': lr_scheduler.state_dict()},
				path)
	print("Save model in results/epoch{}.pth".format(epoch))

def load_model(file, model, optim, lr_scheduler):
	pretrained = torch.load(file)
	model.load_state_dict(pretrained['model'])
	optim.load_state_dict(pretrained['optim'])
	lr_scheduler.load_state_dict(pretrained['lr_scheduler'])
	print("Load model from {}".format(file))

def main(args):

	device = args.device

	model = clip_a2v(args)
	model.to(device)

	optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

	dataset_train = dataset('train')
	dataset_eval  = dataset('eval')

	sampler_train = torch.utils.data.RandomSampler(dataset_train)
	batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
	sampler_eval = torch.utils.data.SequentialSampler(dataset_eval)

	data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train)
	data_loader_eval = DataLoader(dataset_eval, batch_size=1, sampler=sampler_eval, drop_last=False)

	lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=len(data_loader_train)*args.epochs, eta_min=1e-6)

	if args.pretrained:
		load_model(args.pretrained, model, optim, lr_scheduler)

	if args.eval:
		evaluate(model, data_loader_eval, device, len(dataset_eval))
		return

	for epoch in range(args.epochs):
		train(epoch, model, optim, lr_scheduler, data_loader_train, device, int(len(dataset_train) / args.batch_size + 0.5))
		save_model(epoch, model, optim, lr_scheduler)
		score = evaluate(model, data_loader_eval, device, len(dataset_eval))
		path = os.path.join("results", "score" + str(epoch) + ".txt")
		with open(path, 'w') as file:
			file.write(json.dumps(score))

if __name__ == "__main__":
	parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
	args = parser.parse_args()
	main(args)