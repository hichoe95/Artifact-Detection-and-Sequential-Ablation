from config import parse_args
from model import pggan, stylegan2
from misc import print_two_images, print_images
from seq_abl import sequential_ablation

import torch
import numpy as np

torch.manual_seed(45)
np.random.seed(45)

def main(args):

	if args.model == 'stylegan2':
		G, _ = stylegan2(path = args.weight_path, res = args.resolution)
	elif args.model == 'pggan':
		G, _ = pggan(path = args.weight_path, res = args.resolution)
	else:
		print("You can choose pggan or stylegan2")

	device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

	G.to(device)

	sq = sequential_ablation(G, device, args)

	if args.correction:
		original_image, repaired_image, mask_idx = sq.seq_abl(sample_idx = [1,34,123,341], layer_idx= [0,1,3], rate = '30', under = True)
		masks = []
		for i in range(len(mask_idx)):
		    masks.append(image_mask(layer_idx = [0,1,3], act_idx = mask_idx[i], resolution = args.resolution, model = args.model))
		for i in range(original_image.size(0)):
	    	print_two_images(original_image[[i]], repaired_image[[i]], masks[i], labels = ['origin', 'corrected'])
	

	if args.detection:
		norm, arti = sq.arti_detection([0,1,3], sample_num = 15000, topn = 60)
		print_images(norm)
		print_images(arti)

if __name__ == '__main__':
	args = parse_args()

	main(args)
