import pandas as pd
import numpy as np

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import utils
from my_models import regression_ViT as ViT



training_df = pd.read_csv(<INSERT PATH TO CSV HERE>)
target_df = pd.read_csv(<INSERT PATH TO CSV HERE>)

test_df = pd.read_csv(<INSERT PATH TO CSV HERE>)
test_target_df = pd.read_csv(<INSERT PATH TO CSV HERE>)


#_________________________PARAMETERS___________________________
BATCH_SIZE = 100
EPOCHS = 100
LEARNING_RATE = 0.0001
INSTALLED_POWER = 17500

#__________________________MODEL_____________________________
model = ViT(image_size=9, 
            patch_size=3, 
            channels=8,
            dim=64, 
            depth=6, 
            heads=8, 
            mlp_dim=128)



coordenates_of_interest = ['prediction date',

                            '(43.875, -8.375)', '(43.75, -8.375)', '(43.625, -8.375)', '(43.5, -8.375)', '(43.375, -8.375)', '(43.25, -8.375)', '(43.125, -8.375)', '(43.0, -8.375)', '(42.875, -8.375)',

                            '(43.875, -8.25)', '(43.75, -8.25)', '(43.625, -8.25)', '(43.5, -8.25)', '(43.375, -8.25)', '(43.25, -8.25)', '(43.125, -8.25)', '(43.0, -8.25)', '(42.875, -8.25)',
                            
                            '(43.875, -8.125)', '(43.75, -8.125)', '(43.625, -8.125)', '(43.5, -8.125)', '(43.375, -8.125)', '(43.25, -8.125)', '(43.125, -8.125)', '(43.0, -8.125)', '(42.875, -8.125)',

                           '(43.875, -8.0)', '(43.75, -8.0)', '(43.625, -8.0)', '(43.5, -8.0)', '(43.375, -8.0)', '(43.25, -8.0)', '(43.125, -8.0)', '(43.0, -8.0)', '(42.875, -8.0)',
                        
                           '(43.875, -7.875)','(43.75, -7.875)','(43.625, -7.875)','(43.5, -7.875)','(43.375, -7.875)','(43.25, -7.875)','(43.125, -7.875)','(43.0, -7.875)','(42.875, -7.875)',

                            '(43.875, -7.75)','(43.75, -7.75)','(43.625, -7.75)','(43.5, -7.75)','(43.375, -7.75)','(43.25, -7.75)','(43.125, -7.75)','(43.0, -7.75)','(42.875, -7.75)',

                            '(43.875, -7.625)','(43.75, -7.625)','(43.625, -7.625)','(43.5, -7.625)','(43.375, -7.625)','(43.25, -7.625)','(43.125, -7.625)','(43.0, -7.625)','(42.875, -7.625)',

                            '(43.875, -7.5)','(43.75, -7.5)','(43.625, -7.5)','(43.5, -7.5)','(43.375, -7.5)','(43.25, -7.5)','(43.125, -7.5)','(43.0, -7.5)','(42.875, -7.5)',

                            '(43.875, -7.375)','(43.75, -7.375)','(43.625, -7.375)','(43.5, -7.375)','(43.375, -7.375)','(43.25, -7.375)','(43.125, -7.375)','(43.0, -7.375)','(42.875, -7.375)',
                            
                           ]

channels = ['10u', '10v', '2t', 'sp', '100u', '100v', 'vel10_', 'vel100']

#_________________________TRAINING DATA___________________________
labels, data = utils.get_data_and_target_of_interest(training_df, target_df, coordenates_of_interest, channels)
trainset = torch.utils.data.TensorDataset(data, labels)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

#_________________________TEST DATA___________________________
test_labels, test_data = utils.get_data_and_target_of_interest(test_df, test_target_df, coordenates_of_interest, channels)
testset = torch.utils.data.TensorDataset(test_data, test_labels)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)


#_________________________TRAINING THE MODEL___________________________
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

n_total_steps = len(trainloader)

# train the network
for epoch in range(EPOCHS):  # loop over the dataset multiple times

    for i, data in enumerate(trainloader, 0):
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        labels = labels.unsqueeze(1).float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
     
        # verbosity 
        if (i+1)%10000 == 0:
            print(f'epoch {epoch+1}/{EPOCHS}; step {i+1}/{n_total_steps}, loss={loss.item():.4f}') 
       
print('Finished Training')

#_________________________TESTING THE MODEL___________________________
total_samples = len(testloader.dataset)
total_loss = 0
total_loss2 = 0

criterion2 = nn.MAELoss()

with torch.no_grad():
    for data, target in testloader:
        
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
