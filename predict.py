# Imports here
import torch
import json
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch import nn, optim, utils
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser

"""
Helper Functions
"""
class DenseNet_Helper():
	def __init__(self, hidden_units=512, classifier=None):
		self.name = 'DenseNet'
		self.model = models.densenet121(pretrained=True)

		self.classifier = nn.Sequential(
			nn.Linear(1024, hidden_units),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(hidden_units, 256),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(256, 102),  # 102 labels returned from CSV
			nn.LogSoftmax(dim=1)
		) if classifier is None else classifier

	def set_classifier(self, model):
		model.classifier = self.classifier
		return model

	def get_optimizer_parameters(self, model):
		return model.classifier.parameters()

	def get_state_dict(self, model):
		return model.classifier.state_dict()


class ResNet_Helper():
	def __init__(self, hidden_units=512, classifier=None):
		self.name = 'ResNet'
		self.model = models.resnet18(pretrained=True)
		self.classifier = nn.Sequential(
			nn.Linear(512, hidden_units),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(hidden_units, 128),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(128, 102),  # 102 labels returned from CSV
			nn.LogSoftmax(dim=1)
		) if classifier is None else classifier

	def set_classifier(self, model):
		model.fc = self.classifier
		return model

	def get_optimizer_parameters(self, model):
		return model.fc.parameters()

	def get_state_dict(self, model):
		return model.fc.state_dict()


def process_image(image):
	''' Scales, crops, and normalizes a PIL image for a PyTorch model,
		returns an Numpy array
	'''
	im = Image.open(image)
	width, height = im.size

	# Find out which side is smallest and set it to 256
	# Set the other side to super max high value so when we crop we can be sure that smaller side is 256
	if width < height:
		max_width = 256
		max_height = height * 9999
	else:
		max_width = width * 9999
		max_height = 256

	im.thumbnail([max_width, max_height])
	width, height = im.size

	# Crop out center
	# Reference: https://stackoverflow.com/a/46944232/975592
	cropped_img = im.crop(((width - 224) // 2, (height - 224) // 2, (width + 224) // 2, (height + 224) // 2))

	# Convert to numpy array:
	np_image = np.array(cropped_img)

	# Normalize for color channel values
	np_image = np_image / 255
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	np_image -= mean
	np_image /= std

	np_image = np_image.transpose((2, 0, 1))

	# TO DO / CONTINUE ON THIS
	return np_image


def predict(image_path, checkpoint, topk, gpu=False, category_mapping=None):
	''' Predict the class (or classes) of an image using a trained deep learning model.
	'''
	device = torch.device("cuda:0" if torch.cuda.is_available() and gpu==True else "cpu")

	# TODO: Implement the code to predict the class from an image file
	predict_model, optimizer, class_to_idx = load_checkpoint(checkpoint)
	predict_model.to(device)

	# Switch to eval mode
	predict_model.eval()

	# Reference: https://stackoverflow.com/a/483833/975592
	idx_to_class = {v: k for k, v in class_to_idx.items()}

	x = process_image(image_path)
	x = torch.from_numpy(x).float().to(device)

	# Reference: https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612/2
	x.unsqueeze_(0)

	# Calculate the class probabilities (softmax) for img
	with torch.no_grad():
		output = predict_model.forward(x)

	ps = torch.exp(output)
	top_ps, top_classes = ps.topk(topk)

	# Convert numpy array to list and then grab the first (and only) item from the list
	# Note: added cpu() to explicitly copy to host memory if using a GPU to process
	top_ps = top_ps.cpu().numpy()[0]

	top_classes = top_classes.cpu().numpy()[0]
	top_classes = [idx_to_class[idx] for idx in top_classes]

	if category_mapping is not None:
		top_classes = [category_mapping[idx] for idx in top_classes]

	return top_ps.tolist(), top_classes

def load_checkpoint(filepath, gpu=False):
	device = "cuda:0" if torch.cuda.is_available() and gpu==True else "cpu"
	checkpoint = torch.load(filepath, map_location=device)

	model_helper = False
	if 'densenet' in checkpoint['model_name']:
		model_helper = DenseNet_Helper(classifier=checkpoint['classifier'])
	elif 'resnet' in checkpoint['model_name']:
		model_helper = ResNet_Helper(classifier=checkpoint['classifier'])

	if not model_helper:
		raise('Model name ' + str(checkpoint['model_name']) + ' is not valid. Aborting.')

	model = model_helper.model

	# Freeze parameters since we are using a pre-trained network and do not need back propagation
	for param in model.parameters():
		param.requires_grad = False

	# Load the state dict
	# Loop taken from: https://discuss.pytorch.org/t/transfer-learning-missing-key-s-in-state-dict-unexpected-key-s-in-state-dict/33264/3
	state_dict = checkpoint['state_dict']
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = k[7:]  # remove 'module.' of DataParallel
		new_state_dict[name] = v

	model.load_state_dict(new_state_dict, strict=False)
	model = model_helper.set_classifier(model)

	optimizer = optim.Adam(model_helper.get_optimizer_parameters(model), lr=checkpoint['learning_rate'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	return model, optimizer, checkpoint['class_to_idx']

def print_table(row_names, data):
	row_format = "{:>15}" * (len(row_names) + 1)
	print(row_format.format("", *row_names))
	for class_name, row in zip(row_names, data):
		print(row_format.format(class_name, *row))

parser = ArgumentParser()
parser.add_argument('input',
					help='Path to image to be analyzed.')
parser.add_argument('checkpoint', default='model_checkpoint.pth',
					help='Path to checkpoint file.')
parser.add_argument('--top_k', default=5,
					help='Number of top predicted classes to output.')
parser.add_argument('--category_names', default='cat_to_name.json',
					help='Category mapping. Format should be JSON.')
parser.add_argument("--gpu", dest="gpu", default=False,
					help="Use GPU for training", action="store_true")
parser.add_argument("--top_k_print", dest="top_k_print", default=False,
					help="Print out list of top predictions", action="store_true")
args = parser.parse_args()

# Parse command line arguments
path_to_image = args.input
checkpoint = args.checkpoint
top_k = int(args.top_k)
top_k_print = args.top_k_print

with open(args.category_names, 'r') as f:
	category_names = json.load(f)

gpu_available = args.gpu

probs, classes = predict(path_to_image, checkpoint, top_k, gpu=gpu_available, category_mapping=category_names)
high_value_index = probs.index(max(probs))

print(f'=====')
print(f'Top Prediction (with probability')
print(f'=====')
print(f'{classes[high_value_index]} ({probs[high_value_index]*100:.2f}%)')

if top_k_print == True:
	print('\n')
	print(f'========')
	print(f'Top {top_k} Predictions (with probabilities)')
	print(f'========')
	for i in range(len(probs)):
		print(f'{classes[i]} ({probs[i]*100:.2f}%)')