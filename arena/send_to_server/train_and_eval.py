import pandas as pd
import numpy as np

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler

import utils
from my_models import regression_ViT as ViT

from datetime import datetime
import warnings
import os

#_________________________PARAMETERS___________________________

INSTALLED_POWER = 17500

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

IMAGE_SIZE = 9 
PATCH_SIZE = 3
CHANNELS = 8
DIM = 64
DEPTH = 2       # number of transformer blocks
HEADS = 2
MLP_DIM = 512    


#_________________________LOCAL_____________________________
#PATH = './data/'
#
#train_df = pd.read_csv(PATH+'EjemploDatos.csv')
#train_target_df = pd.read_csv(PATH+'EjemploTarget.csv', header=None)
#
#test_df = pd.read_csv(PATH+'EjemploDatos.csv')
#test_target_df = pd.read_csv(PATH+'EjemploTarget.csv', header=None)


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

    #_________________DATA LOADED INTO MEMORY_____________________________

    # TAINING DATA
    df_2016 = pd.read_csv('data_stv_2016.csv')
    df_2017 = pd.read_csv('data_stv_2017.csv')

    meta_data = pd.concat([df_2016, df_2017], axis=0)

    # feature scale the data ignoring the first column
    scaler = StandardScaler()
    meta_data.iloc[:, 1:] = scaler.fit_transform(meta_data.iloc[:, 1:])

    # TRAINING TARGET
    target_2016 = pd.read_csv('target_stv_2016.csv', header=None)
    target_2017 = pd.read_csv('target_stv_2017.csv', header=None)

    meta_target = pd.concat([target_2016, target_2017], axis=0)

    # TEST DATA
    df_2018 = pd.read_csv('data_stv_2018.csv')

    # feature scale the data ignoring the first column
    df_2018.iloc[:, 1:] = scaler.transform(df_2018.iloc[:, 1:])

    # TEST TARGET
    target_2018 = pd.read_csv('target_stv_2018.csv', header=None)

    # warning if meta_data and meta_target have different number of rows
    if meta_data.shape[0] != meta_target.shape[0]:
        warnings.warn('meta_data has {} rows and meta_target has {} rows'.format(meta_data.shape[0], meta_target.shape[0]), RuntimeWarning)


    data, labels = utils.get_data_and_target(meta_data, meta_target, coordinates_of_interest, channels, normalize_target=False)
    trainset = torch.utils.data.TensorDataset(data, labels)

    test_data, test_labels = utils.get_data_and_target(df_2018, target_2018, coordinates_of_interest, channels, normalize_target=False)
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
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#__________________________MODEL_____________________________
model = ViT(image_size=IMAGE_SIZE, # according to the coordinates of interest 
            patch_size=PATCH_SIZE, 
            channels=CHANNELS,   # according to the channels chosen
            dim=DIM, 
            depth=DEPTH, 
            heads=HEADS, 
            mlp_dim=MLP_DIM)

model = model.to(device)

#_________________________TRAINING THE MODEL___________________________
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

n_total_steps = len(trainloader)

# train the network
for epoch in range(EPOCHS):  # loop over the dataset multiple times

    print('Starting epoch: ', epoch+1, '/', EPOCHS, '...')

    for i, data in enumerate(trainloader, 0):
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        labels = labels.unsqueeze(1).float()

        # zero the parameter gradients
        optimizer.zero_grad()

        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward + backward + optimize
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # add nan loss warning and break
        with torch.no_grad():
            if torch.isnan(loss):
                print('Loss is nan. Stopping training.')
                break

        # verbosity 
        if (i+1)%(n_total_steps//4) == 0:
            print(f'EPOCH {epoch+1}/{EPOCHS}; ITERATION {i+1}/{n_total_steps}, BATCH_NMAE={loss.item()/INSTALLED_POWER:.4f}') 
            

                    


print('Finished Training')

print(output)

#_________________________TESTING THE MODEL___________________________
total_samples = len(testloader.dataset)
total_loss = 0
total_loss2 = 0

criterion2 = nn.MSELoss()

print('')
print('Testing the model...')
model.eval()
with torch.no_grad():
    for data, target in testloader:

        data = data.to(device)
        target = target.to(device)
        
        output = model(data)

        loss = criterion(output, target.unsqueeze(1).float())
        total_loss += loss.item()/INSTALLED_POWER

        loss2 = criterion2(output, target.unsqueeze(1).float())
        total_loss2 += loss2.item()/(INSTALLED_POWER**2)


# normalized mean absolute error
nmae = total_loss
print(f'NMAE: {nmae:.4f}')

# normalized mean squared error
nmse = total_loss2
print(f'NMSE: {nmse:.4f}')

ans = input('Do you want to create a file with the results and characteristics of the model?')
if ans=='y':
    # write evaluation results to file
    date_string = datetime.now().strftime("%m_%d-%I_%M_%p")
    with open('{}_img{}_ptch{}_{}.txt'.format(model.__class__.__name__,
                                            IMAGE_SIZE,
                                            PATCH_SIZE,
                                            date_string), 'w') as f:
                
        f.write('TRAINING PARAMETERS: \n')
        f.write(f'Batch size: {BATCH_SIZE} Learning rate: {LEARNING_RATE} Epochs: {EPOCHS} \n')

        f.write('HYPERPARAMETERS: \n')
        f.write(f'Image size: {IMAGE_SIZE} \n Patch size: {PATCH_SIZE} \n Channels: {CHANNELS} \n Dim: {DIM} \n Depth: {DEPTH} \n Heads: {HEADS} \n MLP dim: {MLP_DIM} \n')
        f.write('\n')
        f.write('EVALUATION RESULTS: \n')
        f.write(f'NMAE: {nmae:.4f} \n')
        f.write(f'NMSE: {nmse:.4f} \n')


