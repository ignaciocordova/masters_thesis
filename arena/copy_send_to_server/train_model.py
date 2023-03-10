import pandas as pd
import numpy as np

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

import utils
from my_models import regression_ViT as ViT

import warnings
import os

#_________________________PARAMETERS___________________________
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 0.001
INSTALLED_POWER = 17500

IMAGE_SIZE = 9
PATCH_SIZE = 3


#_________________DATA LOADED INTO MEMORY_____________________________

#df_2016 = pd.read_csv('data_stv_2016.csv')
#df_2017 = pd.read_csv('data_stv_2017.csv', header=None)
#df_2018 = pd.read_csv('data_stv_2018.csv', header=None)
#
#meta_data = pd.concat([df_2016, df_2017, df_2018], axis=0)
#
#
#target_2016 = pd.read_csv('target_stv_2016.csv', header=None)
#target_2017 = pd.read_csv('target_stv_2017.csv', header=None)
#target_2018 = pd.read_csv('target_stv_2018.csv', header=None)
#
#meta_target = pd.concat([target_2016, target_2017, target_2018], axis=0)
#
## warning if meta_data and meta_target have different number of rows
#if meta_data.shape[0] != meta_target.shape[0]:
    #warnings.warn('meta_data has {} rows and meta_target has {} rows'.format(meta_data.shape[0], meta_target.shape[0]), RuntimeWarning)
#
## randomly split meta_data and meta_target into train and test
#train_df, test_df, train_target_df, test_target_df = train_test_split(meta_data, meta_target, test_size=0.2, random_state=42)


#_________________________LOCAL_____________________________
PATH = './data/'

train_df = pd.read_csv(PATH+'EjemploDatos.csv')
train_target_df = pd.read_csv(PATH+'EjemploTarget.csv', header=None)

test_df = pd.read_csv(PATH+'EjemploDatos.csv')
test_target_df = pd.read_csv(PATH+'EjemploTarget.csv', header=None)


#_________________________DATA OF INTEREST_____________________________
coordinates_of_interest = ['prediction date',

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


#______________DATA________________

# ask the user if he wants to create the train and test sets or load them from disk
answer = input('Do you want to create the train and test sets? (y/n)')

if answer == 'y':
    data, labels = utils.get_data_and_target(train_df, train_target_df, coordinates_of_interest, channels, normalize_target=False)
    trainset = torch.utils.data.TensorDataset(data, labels)

    test_data, test_labels = utils.get_data_and_target(test_df, test_target_df, coordinates_of_interest, channels, normalize_target=False)
    testset = torch.utils.data.TensorDataset(test_data, test_labels)

    # create the directory if it doesn't exist
    if not os.path.exists('./processed_data'):
        os.makedirs('./processed_data')
    
    # save in disk
    torch.save(trainset, './processed_data/trainset.pt')
    torch.save(testset, './processed_data/testset.pt')
else:
    # read from disk
    trainset = torch.load('./processed_data/trainset.pt')
    testset = torch.load('./processed_data/testset.pt')

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

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

#_________________________TRAINING THE MODEL___________________________
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

n_total_steps = len(trainloader)

# train the network
for epoch in range(EPOCHS):  # loop over the dataset multiple times

    print('Starting epoch: ', epoch+1, '/', EPOCHS, '...')

    for i, data in enumerate(trainloader, 0):
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        labels = labels.unsqueeze(1).float()
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
     
        # verbosity 
        if (i+1)%1 == 0:
            print(f'EPOCH {epoch+1}/{EPOCHS}; ITERATION {i+1}/{n_total_steps}, LOSS={loss.item():.4f}') 
       
print('Finished Training')



# save trained model in disk
torch.save(model.state_dict(), './trained_model.pt')

