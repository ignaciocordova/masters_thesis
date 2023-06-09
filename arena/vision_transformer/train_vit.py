import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# si quieres aumentar las coordenadas tendras que calcular las nuevas 
# dimensiones de la imagen (de 8 canales)
MAX_LAT = 43.875
MIN_LAT = 42.875
MAX_LON = -7.375
MIN_LON = -8.375

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0001

IMAGE_SIZE = 9 
PATCH_SIZE = 3
CHANNELS = 8
DIM = 64
DEPTH = 4       # number of transformer blocks
HEADS = 2
MLP_DIM = 64    


#_________________________DATA OF INTEREST_____________________________
coordinates_of_interest = ['prediction date']

for lat in np.arange(MAX_LAT, MIN_LAT-0.125, -0.125):
    for lon in np.arange(MIN_LON, MAX_LON+0.125, 0.125):
        coordinates_of_interest.append(f'({lat}, {lon})')

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


    data, labels = utils.get_data_and_target(meta_data, meta_target, coordinates_of_interest, channels, normalize_target=True)
    trainset = torch.utils.data.TensorDataset(data, labels)

    test_data, test_labels = utils.get_data_and_target(df_2018, target_2018, coordinates_of_interest, channels, normalize_target=True)
    testset = torch.utils.data.TensorDataset(test_data, test_labels)

    # create the directory if it doesn't exist
    if not os.path.exists('./images'):
        os.makedirs('./images')
    
    # save in disk
    torch.save(trainset, './images/trainset.pt')
    torch.save(testset, './images/testset.pt')
else:
    # read from disk
    trainset = torch.load('./images/trainset.pt')
    testset = torch.load('./images/testset.pt')

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#__________________________MODEL_____________________________
model = ViT(image_size=IMAGE_SIZE, # according to the coordinates of interest 
            patch_size=PATCH_SIZE, 
            channels=CHANNELS,   # according to the channels chosen
            dim=DIM, 
            depth=DEPTH, 
            heads=HEADS, 
            mlp_dim=MLP_DIM).to(device)

parameters = filter(lambda p: p.requires_grad, model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) 
print('Trainable Parameters:', parameters)

#_________________________TRAINING AND TESTING THE MODEL___________________________

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
eval_losses = []

n_batches = len(testloader)

# train and evaluate the network
for epoch in range(EPOCHS):

    # Training
    model.train()
    train_loss = 0
    for i, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.unsqueeze(1).float().to(device)

        optimizer.zero_grad()
        
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if torch.isnan(loss):
            print('Loss is nan. Stopping training.')
            break

    train_loss /= (len(testloader))
    train_losses.append(train_loss)         

    # Evaluation
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.unsqueeze(1).float().to(device)
            output = model(inputs)
            eval_loss += criterion(output, labels).item()

    eval_loss /= n_batches
    eval_losses.append(eval_loss)

    print(f'EPOCH {epoch+1}/{EPOCHS}, TRAIN_LOSS={train_loss:.4f}, EVAL_LOSS={eval_loss:.4f}')

date_string = datetime.now().strftime("%m_%d-%I_%M_%p")

print('Finished Training')

# plot losses
plot = input("Do you want to plot the losses? (y/n)")
if plot == "y":
    plt.plot(train_losses, label='Training Loss')
    plt.plot(eval_losses, label='Evaluation Loss')
    plt.legend()
    # save figure in a document with the name of the model and the date
    plt.savefig('./figures//LOSSES_{}_img{}_ptch{}_dpth{}_hds{}_{}.png'.format(model.__class__.__name__,
                                            IMAGE_SIZE,
                                            PATCH_SIZE,
                                            DEPTH,
                                            HEADS,
                                            date_string))
    
#_________________________EVALUATION OF THE MODEL___________________________

eval = input("Do you want to evaluate the model? (y/n)")
if eval == "n":
    exit()

total_loss = 0
total_loss2 = 0

criterion2 = nn.MSELoss()

print('')
print('Evaluating the model...')
model.eval()
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
nmae = total_loss/n_batches
print(f'NMAE: {nmae:.4f}')

# normalized mean squared error
nmse = total_loss2/n_batches
print(f'NMSE: {nmse:.4f}')

ans = input('Do you want to save results and characteristics of the model? (y/n)')
if ans=='y':
    # write evaluation results to file
    with open('./results/{}_img{}_ptch{}_dpth{}_hds{}_{}.txt'.format(model.__class__.__name__,
                                            IMAGE_SIZE,
                                            PATCH_SIZE,
                                            DEPTH,
                                            HEADS,
                                            date_string), 'w') as f:
                
        f.write('TRAINING DATASET: \n')
        f.write(f'Number of samples: {len(trainset)} \n')
                
        f.write('TRAINING PARAMETERS: \n')
        f.write(f'Batch size: {BATCH_SIZE} Learning rate: {LEARNING_RATE} Epochs: {EPOCHS} \n')
        f.write(f'Trainable parameters:{parameters}')

        f.write('HYPERPARAMETERS: \n')
        f.write(f'Lat and Lon {MAX_LAT},{MIN_LAT}; {MAX_LON},{MIN_LON} \n')
        f.write(f'Image size: {IMAGE_SIZE} \n Patch size: {PATCH_SIZE} \n Channels: {CHANNELS} \n Dim: {DIM} \n Depth: {DEPTH} \n Heads: {HEADS} \n')
        f.write('\n')
        f.write('EVALUATION RESULTS: \n')
        f.write(f'NMAE: {nmae:.4f} \n')
        f.write(f'NMSE: {nmse:.4f} \n')


# option to save the model 
ans = input('Do you want to save the model? (y/n)')
if ans=='y':
    torch.save(model, './models/vit.pt')


