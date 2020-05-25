import torchvision
import torchvision.transforms as transforms
from lookahead_pytorch import Lookahead
from CIFAR.model import ResNet18
import torch 
from torch import nn 
import numpy as np 
import torchbearer

NB_EPOCHS = 10

torch.manual_seed(3)

# Data setup 
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

channel_means = [np.mean(trainset.data[:,:,:,i]) for i in range(3)]
channel_stds = [np.std(trainset.data[:,:,:,i]) for i in range(3)]

# Transforms
train_transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[x / 255.0 for x in channel_means],
                        std=[x / 255.0 for x in channel_stds])])

test_transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize(mean=[x / 255.0 for x in channel_means],
                        std=[x / 255.0 for x in channel_stds])])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

# Data loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
valloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

device = "cuda:0"
results = []

def train(optimizer_name):

    scheduler = torchbearer.callbacks.torch_scheduler.MultiStepLR(milestones=[60, 120, 160], gamma=0.2)
    model = ResNet18()
    checkpoint = torchbearer.callbacks.ModelCheckpoint('CIFAR\\' + optimizer_name + '_checkpoint.pt')
    logger = torchbearer.callbacks.CSVLogger('CIFAR\\' + optimizer_name + '_log.pt', separator=',', append=True)

    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.001)
    elif optimizer_name == 'Lookahead':
        optimizer = Lookahead(torch.optim.SGD(model.parameters(), lr=0.1), la_alpha=0.8, la_steps=5)
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1)

    loss_function = nn.CrossEntropyLoss()
    trial = torchbearer.Trial(model, optimizer, loss_function, metrics=['loss', 'accuracy'], callbacks=[scheduler, checkpoint, logger]).to(device)
    trial.with_generators(trainloader, val_generator=valloader)
    results.append(trial.run(epochs=NB_EPOCHS))   

# Run 
optimizer_names = ['SGD', 'Lookahead', 'AdamW']
for opt in optimizer_names:
    train(opt)

# Test Plot
import matplotlib.pyplot as plt 
import pandas as pd 

optimizer_names = ['SGD', 'Lookahead', 'AdamW']
plt.figure()
for opt_name, result in zip(optimizer_names, results):
    plt.plot(pd.DataFrame(result)['val_loss'], label=opt_name)
    # pd.DataFrame(result).to_csv("results_"+opt_name)
plt.savefig('CIFAR\\loss_plot.png')

