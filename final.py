from __future__ import print_function, division
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import cv2
import os
import glob
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		
        self.fc1 = nn.Linear(64*12*12, 1024)
        self.drop_layer1 = nn.Dropout(p=0.25)
        
        self.fc2 = nn.Linear(1024, 512)
        self.drop_layer2 = nn.Dropout(p=0.25)
        
        self.fc3 = nn.Linear(512, 2)
        # self.fc2 = nn.Linear(2048, 1024)
        # self.drop_layer3 = nn.Dropout(p=0.25)

    def forward(self, xb):
    	xb = F.relu(self.conv1(xb))
    	xb = self.pool1(xb)
    	xb = F.relu(self.conv2(xb))
    	xb = self.pool2(xb)
    	xb = F.relu(self.conv3(xb))
    	xb = self.pool3(xb) 

    	xb = xb.reshape(xb.shape[0], 64*12*12)
    	
    	xb = F.relu(self.fc1(xb))
    	xb = self.drop_layer1(xb)
    	xb = F.relu(self.fc2(xb))
    	xb = self.drop_layer2(xb)
    	xb = self.fc3(xb)
    	
    	return xb

def train_model(model, criterion, optimizer, num_epochs=25):
	since = time.time()
	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)
		
		for phase in ['train', 'val']:
			if phase == "train":
				model.train()  # Set model to training mode
			else:
				model.eval()   # Set model to evaluate mode
			
			running_loss = 0.0
			running_corrects = 0

			for inputs, labels in dataloaders[phase]:
				inputs = inputs.to(device)
				labels = labels.to(device)
				optimizer.zero_grad()

				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs)
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)\

					if phase == 'train':
						loss.backward()
						optimizer.step()

				# statistics
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]

			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
			# deep copy the model
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
	    time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model

if __name__ == "__main__":

	data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5 ], [0.5, 0.5, 0.5])
    ]),
	}

	data_dir = 'cg_pg'
	image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=100, shuffle=True, num_workers=0) for x in ['train', 'val']}
	dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
	class_names = image_datasets['train'].classes
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	# inputs, classes = next(iter(dataloaders['train']))
	# torchvision.utils.save_image(inputs[30], "first.jpg")
	
	# print(inputs.shape)
	# print(torch.min(inputs[30]))
	# print(torch.max(inputs[30]))
	# exit()
	
	bs = 100
	d_prob = 0.25
	lr = 1e-4
	beta_1 = 0.9
	beta_2 = 0.999
	net = Net()

	loss_func = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr = lr)
	writer = SummaryWriter(log_dir = 'runs/bs100lr4', comment = 'bs100lr4')

	trained_model = train_model(net, loss_func, optimizer, 15)
	torch.save(model.state_dict(), "sftp://avnsh@10.107.42.188/home")

	print("starting of training")
	
	for epoch in range(50):
		for i, (xb, yb) in enumerate(dataloaders['train']):
			pred = net(xb)
			print("shape of output of network", pred.shape)
			loss = loss_func(pred, yb)
			print(loss.item())
			writer.add_scalar("batch loss", loss.item(), i)
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
	
	print('Finished Training')
