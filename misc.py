from matplotlib import gridspec
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import numpy as np

# for mapping layer to resolution
stylegan2 = [0,1,None,2,None,3]
pggan = [None, 0, None, 1, None, 2, None, 3]
shape = [(512, 4, 4), (512, 8, 8), (512, 16, 16), (512, 32, 32)]


def minmax(image):
	return (image - image.min())/(image.max() - image.min())


def image_mask(layer_idx, act_idx, resolution, model):

	shape_idx = stylegan2 if model == 'stylegan2' else pggan
	region_mask = [torch.zeros(shape[shape_idx[l]][0] * shape[shape_idx[l]][1] * shape[shape_idx[l]][2]) for l in layer_idx]

	image_mask = []
	for i in range(len(layer_idx)):
	    region_mask[i][act_idx[i]] = 1
	    temp = region_mask[i].view(shape[i]).mean(dim = 0, keepdim = True).unsqueeze(0)
	    image_mask.append(F.upsample(temp, size = (resolution, resolution), mode = 'bilinear').squeeze())

	image_mask_sum = torch.stack(image_mask).mean(dim = 0)

	return image_mask_sum



def print_two_images(image1, image2, mask, labels, figsize = (18, 5)):

	gs = gridspec.GridSpec(1, 3, wspace = 0.0, hspace = 0.0)

	plt.figure(figsize = figsize)
	plt.tight_layout()

	plt.subplot(gs[0,0])
	plt.axis('off')
	plt.imshow(minmax(image1[0].detach().cpu().permute(1,2,0)))
	plt.title(labels[0])

	plt.subplot(gs[0,1])
	plt.axis('off')
	plt.imshow(minmax(image2[0].detach().cpu().permute(1,2,0)))
	plt.title(labels[1])

	plt.subplot(gs[0,2])
	plt.imshow(minmax(image1[0]).detach().cpu().permute(1,2,0))
	plt.imshow(mask, cmap = 'RdBu_r', vmin = 0.03, vmax = 0.14,alpha = 0.8)
	plt.axis('off')
	plt.title('mask')
	plt.colorbar()

	plt.show()


def print_images(images, title, sample_num = 60):

	gs = gridspec.GridSpec(6, 10, wspace = 0., hspace = 0.0)
	plt.figure(figsize = (10 * 4.87, 6*5))
	plt.tight_layout()

	for i in range(6):
		for j in range(10):
			plt.subplot(gs[i, j])
			plt.imshow(minmax(images[i*10 + j]))
			plt.axis('off')
	plt.suptitle(title, fontsize=50)

	plt.show()
