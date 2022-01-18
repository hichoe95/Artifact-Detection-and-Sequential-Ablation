import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stylegan2_generator import *
from models.stylegan2_discriminator import *
from models.pggan_generator import *
from models.pggan_discriminator import *

def stylegan2(path, res):

	G = StyleGAN2Generator(resolution = res)
	D = StyleGAN2Discriminator(resolution = res)

	weight = torch.load(path)

	G.load_state_dict(weight['generator_smooth'])
	D.load_state_dict(weight['discriminator'])

	G.eval()
	D.eval()

	return G, D

def pggan(path, res):

	G = PGGANGenerator(resolution = res)
	D = PGGANDiscriminator(resolution = res)

	weight = torch.load(path)

	G.load_state_dict(weight['generator_smooth'])
	D.load_state_dict(weight['discriminator'])

	G.eval()
	D.eval()

	return G, D