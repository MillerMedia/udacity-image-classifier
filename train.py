import torch
import json
import sys
import torchvision.models as models
from torchvision import datasets, transforms
from torch import nn, optim, utils
from argparse import ArgumentParser

"""
Helper Functions
"""
class DenseNet_Helper():
	def __init__(self, hidden_units=512):
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
		)

	def set_classifier(self, model):
		model.classifier = self.classifier
		return model

	def get_optimizer_parameters(self, model):
		return model.classifier.parameters()

	def get_state_dict(self, model):
		return model.classifier.state_dict()


class ResNet_Helper():
	def __init__(self, hidden_units=512):
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
		)

	def set_classifier(self, model):
		model.fc = self.classifier
		return model

	def get_optimizer_parameters(self, model):
		return model.fc.parameters()

	def get_state_dict(self, model):
		return model.fc.state_dict()


parser = ArgumentParser()
parser.add_argument('data_dir',
					help='Provide data directory. Mandatory argument')
parser.add_argument("--savedir", dest="savedir", default='.',
					help="Destination to save checkpoints")
parser.add_argument("--arch", dest="arch", default='densenet',
					help="Model architecture used for training")
parser.add_argument("--learning_rate", dest="learning_rate", default=0.003,
					help="Learning rate hyperparameter")
parser.add_argument("--epochs", dest="epochs", default=5,
					help="# of epochs to run")
parser.add_argument("--hidden_units", dest="hidden_units", default=512,
					help="Units in hidden layer")
parser.add_argument("--gpu", dest="gpu", default=False,
					help="Use GPU for training", action="store_true")

args = parser.parse_args()

# Parse command line arguments
data_dir = args.data_dir
savedir = args.savedir
arch = args.arch
learning_rate = float(args.learning_rate)
epochs = int(args.epochs)
hidden_units = int(args.hidden_units)
gpu_available = args.gpu

"""
Valid command line arguments
"""

# I keep it broad since
# so the user can enter 'densenet', 'densenet121' or other variations.
# TODO: architecture
if 'densenet' not in arch and 'resnet' not in arch:
	print(
		"This training script currently only supports 'densenet121' or 'resnet18'. Please use 'densenet' or 'resnet' as your '--arch' value.")
	sys.exit()

if 'densenet' in arch:
	model_helper = DenseNet_Helper()
elif 'resnet' in arch:
	model_helper = ResNet_Helper()

print("Using model architecture " + model_helper.name)

if hidden_units >= 1023 or hidden_units <= 257:
	print("Hidden units value must be between 257 and 1023. Please update.")
	sys.exit()

# Subdirectories for data
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define your transforms for the training, validation, and testing sets
means = [0.485, 0.456, 0.406]
stdev = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
	transforms.RandomRotation(45),
	transforms.RandomResizedCrop(224),
	transforms.RandomHorizontalFlip(),
	transforms.RandomVerticalFlip(),
	transforms.ToTensor(),
	transforms.Normalize(means, stdev)
])

valid_train = [
	transforms.Resize(255),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(means, stdev)
]

valid_transforms = transforms.Compose(valid_train)
test_transforms = transforms.Compose(valid_train)

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
validloader = utils.data.DataLoader(valid_data, batch_size=32)
testloader = utils.data.DataLoader(test_data, batch_size=32)

with open('cat_to_name.json', 'r') as f:
	cat_to_name = json.load(f)

model = model_helper.model

# Freeze parameters since we are using a pre-trained network and do not need back propagation
for param in model.parameters():
	param.requires_grad = False

model = model_helper.set_classifier(model)

# Define variables for training process
print_every = 20

# Can test this with GPU on or off with this code
device = torch.device("cuda:0" if torch.cuda.is_available() and gpu_available == True else "cpu")
model.to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model_helper.get_optimizer_parameters(model), lr=learning_rate)

print("Training with " + str(device))
print(f'Size of training set: {len(trainloader)}')

train_losses, valid_losses, train_accuracies, valid_accuracies = [], [], [], []
model.train()

"""
Train the model
"""
for e in range(epochs):
	current_epoch = str(e + 1)
	print(f'Starting epoch: {current_epoch}')

	running_loss = 0
	step = 0
	train_accuracy = 0

	for ii, (inputs, labels) in enumerate(trainloader):
		step += 1
		inputs, labels = inputs.to(device), labels.to(device)

		# Zero gradient out for each step
		optimizer.zero_grad()

		output = model.forward(inputs)
		loss = criterion(output, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()

		# Check if training is getting better...
		train_ps = torch.exp(output)
		top_p, top_class = train_ps.topk(1, dim=1)
		equals = top_class == labels.view(*top_class.shape)
		train_accuracy += torch.mean(equals.type(torch.FloatTensor))

		if step % print_every == 0:
			# Test against validation set
			valid_running_loss = 0
			accuracy = 0

			# Switch to eval mode
			model.eval()

			with torch.no_grad():
				for jj, (inputs, labels) in enumerate(validloader):
					inputs, labels = inputs.to(device), labels.to(device)
					output = model.forward(inputs)
					valid_loss = criterion(output, labels)

					valid_running_loss += valid_loss

					# Calculate accuracy
					ps = torch.exp(output)
					top_p, top_class = ps.topk(1, dim=1)
					equals = top_class == labels.view(*top_class.shape)
					accuracy += torch.mean(equals.type(torch.FloatTensor))

			print(
				f'Epoch {current_epoch} of {epochs} | Train loss: {running_loss / print_every:.3f} | Train acc: {train_accuracy / step:.3f} | Val. loss: {valid_running_loss / len(validloader):.3f} | Val. acc.: {accuracy / len(validloader):.3f}')

			# Switch back to train
			model.train()

			train_losses.append(running_loss / print_every)
			valid_losses.append(valid_running_loss / len(validloader))

			train_accuracies.append(train_accuracy / step * 100)
			valid_accuracies.append(accuracy / len(validloader) * 100)

			running_loss = 0

checkpoint = {
	'model_name': arch,
	'epochs': epochs,
	'learning_rate': learning_rate,
	'classifier': model_helper.classifier,
	'classifier_state_dict': model_helper.get_state_dict(model),
	'state_dict': model.state_dict(),
	'optimizer_state_dict': optimizer.state_dict(),
	'plot_data': {
		'train_accuracies': train_accuracies,
		'valid_accuracies': valid_accuracies,
		'train_losses': train_losses,
		'valid_losses': valid_losses
	},
	'class_to_idx': train_data.class_to_idx,
}

torch.save(checkpoint, 'model_checkpoint.pth')