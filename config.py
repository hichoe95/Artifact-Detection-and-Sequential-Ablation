import os
import argparse


def parse_args(jupyter = False):
	parser = argparse.ArgumentParser()

	parser.add_argument("--gpu", type = int, default = 0, help = "gpu index numper")
	parser.add_argument("--batch_size", type = int, default = 30, help = "batch size for pre processing and generating process")
	parser.add_argument("--sample_size", type = int, default = 30000, help = "sample size for statistics")
	parser.add_argument("--freq_path", type = str, default = "./stats", help = "loading saved frequencies of neurons")
	# parser.add_argument("--freq_path", type = str, default = "", help = "loading saved frequencies of neurons")


	# build model
	parser.add_argument("--model", type = str, default = "stylegan2", help = "pggan, styelgan2")
	parser.add_argument("--dataset", type = str, default = "ffhq", help = "ffhq, cat, church, etc")
	parser.add_argument("--resolution", type = int, default = 1024, help = "dataset resolution")
	parser.add_argument("--weight_path", type = str, default = "./stylegan2_ffhq1024.pth", help = "pre-trained weight path")


	# implementation
	parser.add_argument("--detection", type = bool, default = True, help = "implement normal/artifact detection")
	parser.add_argument("--correction", type = bool, default = True, help = "implement correction task")

	config = parser.parse_args() if jupyter == False else parser.parse_args(args = [])

	return config