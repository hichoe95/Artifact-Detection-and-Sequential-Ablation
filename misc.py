from matplotlib import gridspec
import matplotlib.pyplot as plt

import torch
import numpy as np



def minmax(image):
	return (image - image.min())/(image.max() - image.min())


def print_two_images(image1, image2, labels, figsize = (10, 5)):
	gs = gridspec.GridSpec(1, 2, wspace = 0.1, hspace = 0.1)

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