import pandas as pd
import numpy as np

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

import utils
from my_models import regression_ViT as ViT

import warnings
import os


#_________________________PARAMETERS___________________________
BATCH_SIZE = 32
INSTALLED_POWER = 17500

IMAGE_SIZE = 9
PATCH_SIZE = 3


testset = torch.load('./processed_data/testset.pt')
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#__________________________MODEL_____________________________
model = ViT(image_size=IMAGE_SIZE, # according to the coordinates of interest 
            patch_size=PATCH_SIZE, 
            channels=8,   # according to the channels chosen
            dim=64, 
            depth=4, 
            heads=6, 
            mlp_dim=128)

model = model.to(device)

# load the trained model
model.load_state_dict(torch.load('./trained_model.pt'))

#_________________________TESTING THE MODEL___________________________
total_samples = len(testloader.dataset)
total_loss = 0
total_loss2 = 0

criterion = nn.MSELoss()
criterion2 = nn.L1Loss()

print('')
print('Testing the model...')
with torch.no_grad():
    for data, target in testloader:

        data = data.to(device)
        target = target.to(device)
        
        output = model(data)

        loss = criterion(output, target.unsqueeze(1).float())
        total_loss += loss.item()

        loss2 = criterion2(output, target.unsqueeze(1).float())
        total_loss2 += loss2.item()

# normalized mean absolute error
nmae = total_loss2/(INSTALLED_POWER)
print(f'NMAE: {nmae:.4f}')

# normalized mean squared error
nmse = total_loss/(INSTALLED_POWER**2) 
print(f'NMSE: {nmse:.4f}')

# write evaluation results in file 
with open('./{}_results.txt'.format(model.__class__.__name__), 'w') as f:
    f.write(f'NMAE: {nmae:.4f}, NMSE: {nmse:.4f}')

print(model.__class__.__name__)