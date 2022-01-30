import os

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class sequential_ablation(object):
	def __init__(self, G : nn.Module, device, args):

		self.G = G
		self.device = device
		G.to(self.device)

		self.model = args.model
		self.sample_size = args.sample_size

		# random sampling latent z
		self.latents = torch.randn((self.sample_size, 512))

		self.batch_size = args.batch_size
		self.layers = [0, 1, 3, 5] if self.model == 'stylegan2' else [1, 3, 5, 7]

		# if you have saved .npy file, then it will load it. Otherwise, it should be calculated.
		if args.freq_path != "":
			self.neurons_freq = {f'layer{i:d}' : np.load(os.path.join(args.freq_path,f'rate_layer{i:d}_{args.dataset}_{args.model}.npy')) for i in self.layers}
			print("Frequencies of the neurons are loaded !")
		else:
			print("Calculating frequency of the neurons ...")
			self.neurons_freq = self.neurons_statistic()
			print("Frequencies of the neurons are calculated !")

		self.r_indices, self.r_indices_ = self.index_ratio()
		


	# sequential ablation for given latent codes.
	@torch.no_grad()
	def seq_abl(self, sample_idx : list, layer_idx : list, rate = '30', under = True):

		assert len(sample_idx) <= self.batch_size, "len(sample_idx) should be lower than batch_size"

		if under:
			r_indice = [index[rate] for index in self.r_indices_.values()]
		else:
			r_indice = [index[rate] for index in self.r_indices.values()]

		act_idx = []
		sep = []

		def hook_function(index):
			def fn(_, __, o):
				nonlocal act_idx

				o = o.view(o.size(0), -1) if self.model == 'pggan' else o[0].view(o[0].size(0), -1)

				# for mask
				mask = o[:,index].detach().cpu() > 0
				idx_expand = torch.tensor(index).expand(len(sample_idx),-1)
				act_idx.append(idx_expand[mask])
				sep.append(mask.numpy().sum(axis = 1).cumsum())

				# ablation
				o[:,index] = torch.where(o[:, index] > 0, torch.tensor(0.).to(self.device), o[:, index])

			return fn

		hook = [getattr(self.G if self.model == 'pggan' else self.G.synthesis, 'layer' + str(layer)).register_forward_hook(hook_function(index)) \
	    		 for index, layer in zip(r_indice, layer_idx)]
	    
		repaired_image = self.G(self.latents[sample_idx].to(self.device))['image'].detach().cpu()
	    
		for h in hook:
			h.remove()

		original_image = self.G(self.latents[sample_idx].to(self.device))['image'].detach().cpu()

		# for masking what neurons are turned off.
		mask_idx = []

		for sample in range(len(sample_idx)):
			temp = []
			for layer in range(len(layer_idx)):
				if sample == 0:
					temp.append(act_idx[layer][0 : sep[layer][sample]])
				else:
					temp.append(act_idx[layer][sep[layer][sample - 1] : sep[layer][sample]])
			mask_idx.append(temp)

		return original_image, repaired_image, mask_idx



	# you can detect artifact images and normal images in generated N (sample_num) images.
	@torch.no_grad()
	def arti_detection(self, layer_idx : list, sample_num = 30000, topn = 30, rate = '30'):
		# we use new samples differ from calculating statistics.

		assert sample_num % self.batch_size == 0, 'sample_sum should be devided by batch_size.'

		z = torch.randn((sample_num, 512))

		
		total_list = []

		for i in tqdm(range(sample_num // self.batch_size)):
			r_indice = [index[rate] for index in self.r_indices_.values()]

			temp = torch.zeros(self.batch_size) * 1.

			def hook_function(index):
				def fn(_, __, o):
					nonlocal temp

					if self.model == 'stylegan2':
						mask = o[0].detach().cpu().view(self.batch_size, -1)[:, index] > 0
					elif self.model == 'pggan':
						mask = o.detach().cpu().view(self.batch_size, -1)[:, index] > 0

					temp += mask.sum(dim = -1)

				return fn

			hook = [getattr(self.G if self.model == 'pggan' else self.G.synthesis, 'layer' + str(layer)).register_forward_hook(hook_function(index)) \
					for index, layer in zip(r_indice, layer_idx)]
			images = self.G(z[i * self.batch_size : (i+1) * self.batch_size].to(self.device))
			for h in hook:
				h.remove()

			total_list.append(temp)

		total_list = torch.cat(total_list, dim = 0)
		sorted_index = total_list.numpy().argsort()

		normal = sorted_index[:topn]
		artifact = sorted_index[-topn:]

		# for plotting images
		normal_img = self.generate_images(z[normal])
		artifact_img = self.generate_images(z[artifact])

		return normal_img, artifact_img


	# Calculating the probability of each neuron empitically, since we don't know distribution of neuron frequency for the generator.
	@torch.no_grad()
	def neurons_statistic(self,):

		ratio = {f'layer{i:d}' : None for i in self.layers}

		def function(layer_num):
			def fn(_ ,__, o):

				key = 'layer' + str(layer_num)
				if ratio[key] == None:
					if self.model == 'stylegan2':
						ratio[key] = (o[0].detach().cpu().view(self.batch_size, -1) > 0) * 1.0
					elif self.model == 'pggan':
						ratio[key] = (o.detach().cpu().view(self.batch_size, -1) > 0) * 1.0
				else:
					if self.model == 'stylegan2':
						ratio[key] += (o[0].detach().cpu().view(self.batch_size, -1) > 0) * 1.0
					elif self.model == 'pggan':
						ratio[key] += (o.detach().cpu().view(self.batch_size, -1) > 0) * 1.0

			return fn

		hook = [getattr(self.G if self.model == 'pggan' else self.G.synthesis, 'layer' + str(i)).register_forward_hook(function(i)) \
				for i in self.layers]

		for j in tqdm(range(self.sample_size // self.batch_size)):
			temp = self.G(self.latents[j * self.batch_size : (j + 1) * self.batch_size].to(self.device))

		for h in hook:
			h.remove()

		ratio = {key : val.sum(axis = 0) / (1. * self.sample_size) for key, val in ratio.items()}

		return ratio


	# neurons of indices with activation rate R in each layer.
	def index_ratio(self,):
		r_index = {} # neurons with activation upper rate R
		r_index_ = {} # neurons with activation under rate R

		for k, v in tqdm(self.neurons_freq.items()):
			v = v.flatten()
			temp_r_index = {}
			temp_r_index_ = {}
			for i in range(10,0,-1):
				temp_r_index[str(i * 10)] = np.where(v >= i * 1. / 10)[0]
				temp_r_index_[str(i * 10)] = np.intersect1d(np.where(v <= i * 1. / 10)[0], np.where(v > 0)[0])
			r_index[k] = (temp_r_index)
			r_index_[k] = (temp_r_index_)

		return r_index, r_index_


	@torch.no_grad()
	def generate_images(self, latents : torch.tensor):
		images = []
		for i in range(latents.size(0) // self.batch_size):
			images.append(self.G(latents[i * self.batch_size : (i + 1) * self.batch_size].to(self.device))['image'].permute(0,2,3,1).detach().cpu())

		images = torch.cat(images, dim = 0)

		return images


