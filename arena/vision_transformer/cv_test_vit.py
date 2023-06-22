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

# if you want to increase the geographical scope you will need to 
# calculate the new image dimensions of the enlarged area
MAX_LAT = 43.875
MIN_LAT = 42.875
MAX_LON = -7.375
MIN_LON = -8.375

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0003

IMAGE_SIZE = 9 
PATCH_SIZE = 3
CHANNELS = 8
DIM = 64
DEPTH = 8       # number of transformer blocks
HEADS = 8
MLP_DIM = 64    

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

param_grid = {'depth': [1, 2, 4],
              'heads': [1, 2, 4, 8]}

best_val_loss = 1000

for depth in param_grid['depth']:
    for heads in param_grid['heads']:
        print('---------------------')
        print('Training with depth {} and heads {}'.format(depth, heads))
        print('---------------------')
        model = ViT(image_size=IMAGE_SIZE, # according to the coordinates of interest 
                    patch_size=PATCH_SIZE, 
                    channels=CHANNELS,   # according to the channels chosen
                    dim=DIM, 
                    depth=depth, 
                    heads=heads, 
                    mlp_dim=MLP_DIM).to(device)
    
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # early stopping
        no_improvement = 0
        patience = 10 

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

                # if eval loss is lower than the previous one, save the hyperparameters
                if len(val_losses) == 0 or val_loss < min(val_losses):
                    no_improvement = 0
                else:
                    no_improvement +=1

                if no_improvement >= patience:
                    print('Early stopping in epoch {}'.format(epoch+1))
                    break

                val_losses.append(val_loss)

        print('Train and val losses for depth {} and heads {}:'.format(depth, heads))
        print('Train loss: {:.4f}'.format(train_loss))
        print('Val loss: {:.4f}'.format(val_loss))
        print('---------------------')  

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_depth = depth
            best_heads = heads

print('Best depth and heads: {}, {}'.format(best_depth, best_heads))
print('---------------------')
print('Starting training with best hyperparameters...')
print('---------------------')

DEPTH = best_depth
HEADS = best_heads

# merge train and validation sets
trainset = torch.utils.data.ConcatDataset([trainset, valset])
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

#_________________________TRAINING AND TESTING THE MODEL___________________________


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
print('---------------------')
print('Trainable Parameters:', parameters)
print('---------------------')

train_losses = []
test_losses = []

no_improvement = 0

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
    test_loss = 0
    test_loss2 = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.unsqueeze(1).float().to(device)
            
            output = model(inputs)
            loss = criterion(output, labels)
            loss2 = criterion2(output, labels)


            test_loss += loss.item()
            test_loss2 += loss2.item()
    
        test_loss /= len(testloader)
        test_loss2 /= len(testloader)

        # if eval loss is lower than the previous one, save the model
        if len(test_losses) == 0 or test_loss < min(test_losses):
            nmae = test_loss
            nmse = test_loss2
            print('Saving model at epoch {} with test_loss {}'.format(epoch+1, nmae))
            torch.save(model, './models/vit.pt')
            no_improvement = 0
        else:
            no_improvement +=1
        
        if no_improvement >= patience:
            print('Early stopping in epoch {}'.format(epoch+1))
            break
        test_losses.append(test_loss)

    print(f'EPOCH {epoch+1}/{EPOCHS}, TRAIN_NMAE={train_loss:.4f}, TEST_NMAE={test_loss:.4f}')


# plot losses
plot = input("Do you want to plot the losses? (y/n)")
if plot == "y":
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Evaluation Loss')
    plt.legend()
    # save figure in a document with the name of the model and the date
    plt.savefig('./figures/LOSSES_vit_img{}_ptch{}_dpth{}_hds{}_{}.png'.format(IMAGE_SIZE,
                                                                                PATCH_SIZE,
                                                                                DEPTH,
                                                                                HEADS,
                                                                                date_string))



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
nmae = total_loss/len(testloader)
print(f'NMAE: {nmae:.4f}')

# normalized mean squared error
nmse = total_loss2/len(testloader)
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


