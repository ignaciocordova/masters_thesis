import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler

import utils
from my_models import WindCNN

from datetime import datetime
import warnings
import os

#_________________________PARAMETERS___________________________

INSTALLED_POWER = 17500

# if you want to increase the geographical scope you will need to 
# calculate the new image dimensions of the enlarged area
MAX_LAT = 43.875
MIN_LAT = 42.875
MAX_LON = -7.375
MIN_LON = -8.375

BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.0001

IMAGE_SIZE = 9 
CHANNELS = 8

date_string = datetime.now().strftime("%m_%d-%I_%M_%p")


#_________________________DATA OF INTEREST_____________________________
coordinates_of_interest = ['prediction date']

for lat in np.arange(MAX_LAT, MIN_LAT-0.125, -0.125):
    for lon in np.arange(MIN_LON, MAX_LON+0.125, 0.125):
        coordinates_of_interest.append(f'({lat}, {lon})')

channels = ['10u', '10v', '2t', 'sp', '100u', '100v', 'vel10_', 'vel100']


#______________DATA________________

# ask the user if he wants to create the train and test sets or load them from disk
answer = input('Do you want to create the train, validation and test sets? (y/n)')

if answer == 'y':

    #_________________DATA LOADED INTO MEMORY_____________________________

    # TAINING DATA AND TARGET
    train_data = pd.read_csv('data_stv_2016.csv')
    train_target = pd.read_csv('target_stv_2016.csv', header=None)
    # feature scale the data ignoring the first column
    scaler = StandardScaler()
    train_data.iloc[:, 1:] = scaler.fit_transform(train_data.iloc[:, 1:])


    # VALIDATION DATA AND TARGET
    val_data = pd.read_csv('data_stv_2017.csv')
    val_target = pd.read_csv('target_stv_2017.csv', header=None)
    # feature scale the data ignoring the first column
    val_data.iloc[:, 1:] = scaler.transform(val_data.iloc[:, 1:])

    # TEST DATA AND TARGET
    test_data = pd.read_csv('data_stv_2018.csv')
    test_target = pd.read_csv('target_stv_2018.csv', header=None)
    # feature scale the data ignoring the first column
    test_data.iloc[:, 1:] = scaler.transform(test_data.iloc[:, 1:])


    # warning if meta_data and meta_target have different number of rows
    if train_data.shape[0] != train_target.shape[0]:
        warnings.warn('meta_data has {} rows and meta_target has {} rows'.format(train_data.shape[0], train_target.shape[0]), RuntimeWarning)


    data, labels = utils.get_data_and_target(train_data, train_target, coordinates_of_interest, channels, normalize_target=True)
    trainset = torch.utils.data.TensorDataset(data, labels)

    data, labels = utils.get_data_and_target(val_data, val_target, coordinates_of_interest, channels, normalize_target=True)
    valset = torch.utils.data.TensorDataset(data, labels)

    test_data, test_labels = utils.get_data_and_target(test_data, test_target, coordinates_of_interest, channels, normalize_target=True)
    testset = torch.utils.data.TensorDataset(test_data, test_labels)

    # create the directory if it doesn't exist
    if not os.path.exists('./images'):
        os.makedirs('./images')
    
    # save in disk
    torch.save(trainset, './images/trainset.pt')
    torch.save(valset, './images/valset.pt')
    torch.save(testset, './images/testset.pt')

    # save description of the dataset
    with open('./images/dataset_description.txt', 'w') as f:
        f.write(f'TRAINING DATASET on date {date_string}: \n')
        f.write(f'Number of samples: {len(trainset)} \n')
        f.write('VALIDATION DATASET: \n')
        f.write(f'Number of samples: {len(valset)} \n')
        f.write('TEST DATASET: \n')
        f.write(f'Number of samples: {len(testset)} \n')
        f.write('------------------------------------')
        f.write('Image dimensions: \n')
        f.write(f'Lat and Lon {MAX_LAT},{MIN_LAT}; {MAX_LON},{MIN_LON} \n')
        f.write(f'Image size: {IMAGE_SIZE} \n Channels: {CHANNELS}')

else:
    # read from disk
    trainset = torch.load('./images/trainset.pt')
    valset = torch.load('./images/valset.pt')
    testset = torch.load('./images/testset.pt')

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('---------------------')
print('Using device:', device)
print('---------------------')

#_________________________TRAIN AND VALIDATION___________________________

criterion = nn.L1Loss()
criterion2 = nn.MSELoss()

param_grid = {'num_conv_layers': [4, 10, 20],
              'kernel_size': [1, 3, 5]}

best_val_loss = 10000

for num_conv_layers in param_grid['num_conv_layers']:
    for kernel_size in param_grid['kernel_size']:
        print('---------------------')
        print('Training with depth {} and heads {}'.format(num_conv_layers, kernel_size))
        print('---------------------')
        model = WindCNN(num_channels=CHANNELS,
                        image_width=IMAGE_SIZE,
                        image_height=IMAGE_SIZE,
                        num_conv_layers=num_conv_layers,
                        kernel_size=kernel_size)
    
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # early stopping
        no_improvement = 0
        patience = 15 

        train_losses = []
        val_losses = []

        for epoch in range(EPOCHS):
            # Training
            model.train()
            train_loss = 0
            for i, (inputs, labels) in enumerate(trainloader):
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

            train_loss /= (len(trainloader))
            train_losses.append(train_loss)

            # Evaluation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, labels in valloader:
                    inputs, labels = inputs.to(device), labels.unsqueeze(1).float().to(device)
                    
                    output = model(inputs)
                    loss = criterion(output, labels)
                    val_loss += loss.item()
            
                val_loss /= (len(valloader))

                # if eval loss is lower than the previous one
                if len(val_losses) == 0 or val_loss < min(val_losses):
                    no_improvement = 0
                else:
                    no_improvement +=1

                if no_improvement >= patience:
                    print('---------------------')
                    print('Early stopping in epoch {}'.format(epoch+1))
                    print('Current val loss: {:.4f}'.format(val_loss))
                    print('Optimal val loss: {:.4f}'.format(min(val_losses)))
                    print('Last patience val losses: ', np.round(val_losses[-patience:], 4))
                    print('---------------------')
                          
                    break

                val_losses.append(val_loss)

        print('Train and val losses for num_conv_layers {} and kernel_size {}:'.format(num_conv_layers, kernel_size))
        print('Train loss: {:.4f}'.format(np.round(train_losses[-patience], 4)))
        print('Val loss: {:.4f}'.format(np.round(val_losses[-patience], 4)))
        print('---------------------')  

        if val_losses[-patience] < best_val_loss:
            best_val_loss = val_losses[-patience]
            best_val_losses = val_losses
            best_train_losses = train_losses
            best_num_conv_layers = num_conv_layers
            best_kernel_size = kernel_size
            best_epoch = epoch-patience

EPOCHS = int(best_epoch*1.5)

# plot losses
plot = input("Do you want to plot the losses? (y/n)")
if plot == "y":
    plt.plot(best_train_losses, label='Training Loss')
    plt.plot(best_val_losses, label='Validation Loss')
    plt.legend()
    # save figure in a document with the name of the model and the date
    plt.savefig('./figures/LOSSES_vit_img{}_ptch{}_nconvs{}_kernel{}_{}.png'.format(IMAGE_SIZE,
                                                                                best_num_conv_layers,
                                                                                best_kernel_size,
                                                                                date_string))


# merge train and validation sets
trainset = torch.utils.data.ConcatDataset([trainset, valset])
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

print('Best depth and heads: {}, {}'.format(best_num_conv_layers, best_kernel_size))
print('---------------------')
print('Starting training with best hyperparameters...')
print('---------------------')

#_________________________TRAINING THE MODEL___________________________


#__________________________MODEL_____________________________
model = WindCNN(num_channels=CHANNELS,
                        image_width=IMAGE_SIZE,
                        image_height=IMAGE_SIZE,
                        num_conv_layers=best_num_conv_layers,
                        kernel_size=best_kernel_size)

parameters = filter(lambda p: p.requires_grad, model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) 
print('---------------------')
print('Trainable Parameters:', parameters)
print('---------------------')

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
model.train()

for epoch in range(EPOCHS):    
    train_loss = 0
    for i, (inputs, labels) in enumerate(trainloader):
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

    train_loss /= (len(trainloader))

    print(f'EPOCH {epoch+1}/{EPOCHS}, TRAIN_NMAE={train_loss:.4f}')


#_________________________SAVE MODEL___________________________

# save trained model
torch.save(model.state_dict(), './models/CNN.pt')

#_________________________EVALUATION___________________________

# Evaluation
model.eval()
nmae = 0
nmse = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.unsqueeze(1).float().to(device)
        
        output = model(inputs)
        nmae += criterion(output, labels)
        nmse += criterion2(output, labels)

    nmae /= (len(testloader))
    nmse /= (len(testloader))

print('-------------------')
print('NMAE: ', nmae)
print('NMSE: ', nmse)


ans = input('Do you want to save results and characteristics of the model? (y/n)')
if ans=='y':
    # write evaluation results to file
    with open('./results/{}_img{}_nconvs{}_kernel{}_{}.txt'.format(model.__class__.__name__,
                                                                    IMAGE_SIZE,
                                                                    best_num_conv_layers,
                                                                    best_kernel_size,
                                                                    date_string)):
                
        f.write('TRAINING DATASET: \n')
        f.write(f'Number of samples: {len(trainset)} \n')
                
        f.write('TRAINING PARAMETERS: \n')
        f.write(f'Batch size: {BATCH_SIZE} Learning rate: {LEARNING_RATE} Epochs: {EPOCHS} \n')
        f.write(f'Trainable parameters:{parameters}')

        f.write('HYPERPARAMETERS: \n')
        f.write(f'Lat and Lon {MAX_LAT},{MIN_LAT}; {MAX_LON},{MIN_LON} \n')
        f.write(f'Image size: {IMAGE_SIZE} \n Channels: {CHANNELS} \n Num Convs: {best_num_conv_layers} \n Kernel: {best_kernel_size} \n')
        f.write('\n')
        f.write('EVALUATION RESULTS: \n')
        f.write(f'NMAE: {nmae:.4f} \n')
        f.write(f'NMSE: {nmse:.4f} \n')



