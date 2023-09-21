import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler

import utils
from my_models import ViViT

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

NUM_FRAMES = 4 # look 4 hours to the past
OVERLAP_SIZE = 3 #number of overlaping frames

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0001

IMAGE_SIZE = 9 
PATCH_SIZE = 3
CHANNELS = 8+1 # 8 channels + 1 previous label channel
DIM = 64
DEPTH = 4       # number of transformer blocks
HEADS = 4

N_HOURS = 6 # number of hours to predict into the future

date_string = datetime.now().strftime("%m_%d-%I_%M_%p")

#_________________________DATA OF INTEREST_____________________________
coordinates_of_interest = ['prediction date']

for lat in np.arange(MAX_LAT, MIN_LAT-0.125, -0.125):
    for lon in np.arange(MIN_LON, MAX_LON+0.125, 0.125):
        coordinates_of_interest.append(f'({lat}, {lon})')

channels = ['10u', '10v', '2t', 'sp', '100u', '100v', 'vel10_', 'vel100']


#______________DATA________________

# ask the user if he wants to create the train and test sets or load them from disk
answer = input(f'Create the PAST LABEL INFORMED VIDEO train, validation and test sets {N_HOURS} into the future? (y/n)')

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

    # print first N_HOURS+2 rows of data and target
    print('First {} rows of target:'.format(N_HOURS+2))
    print(train_target.head(N_HOURS+2))

    if N_HOURS>1:
        # drop the last N_HOURS rows of data and the first N_HOURS rows of target
        train_data = train_data.iloc[:-N_HOURS+1, :]
        train_target = train_target.iloc[N_HOURS-1:, :]
        val_data = val_data.iloc[:-N_HOURS+1, :]
        val_target = val_target.iloc[N_HOURS-1:, :]
        test_data = test_data.iloc[:-N_HOURS+1, :]
        test_target = test_target.iloc[N_HOURS-1:, :]

    # print first N_HOURS+2 rows of data and target
    print('First {} rows of target after drop:'.format(N_HOURS+2))
    print(train_target.head(N_HOURS+2))

    # warning if meta_data and meta_target have different number of rows
    if train_data.shape[0] != train_target.shape[0]:
        warnings.warn('meta_data has {} rows and meta_target has {} rows'.format(train_data.shape[0], train_target.shape[0]), RuntimeWarning)

    train_images, train_labels = utils.get_data_and_target_with_previous_label_channel(train_data, train_target, coordinates_of_interest, channels, IMAGE_SIZE, normalize_target=True)
    trainset = torch.utils.data.TensorDataset(train_images, train_labels)

    val_images, val_labels = utils.get_data_and_target_with_previous_label_channel(val_data, val_target, coordinates_of_interest, channels, IMAGE_SIZE, normalize_target=True)
    # first image in val set has label channel of last in train set
    val_images[0, -1, :, :] = trainset[-1][1]
    # add last NUM_FRAMES frames of the trainset to the beggining of the valset
    val_images = torch.cat((train_images[-NUM_FRAMES+1:], val_images), dim=0)
    val_labels = torch.cat((train_labels[-NUM_FRAMES+1:], val_labels), dim=0)
    valset = torch.utils.data.TensorDataset(val_images, val_labels)


    test_images, test_labels = utils.get_data_and_target_with_previous_label_channel(test_data, test_target, coordinates_of_interest, channels, IMAGE_SIZE, normalize_target=True)
    # first image in test set has label channel of last in train set
    test_images[0, -1, :, :] = valset[-1][1]
    # add last NUM_FRAMES frames of the valset to the beggining of the testset
    test_data = torch.cat((val_images[-NUM_FRAMES+1:], test_images), dim=0)
    test_labels = torch.cat((val_labels[-NUM_FRAMES+1:], test_labels), dim=0)
    testset = torch.utils.data.TensorDataset(test_data, test_labels)

    trainloader = DataLoader(trainset, batch_size=1, shuffle=False)
    valloader = DataLoader(valset, batch_size=1, shuffle=False)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    video_trainset = utils.create_video_dataset(trainloader, NUM_FRAMES, OVERLAP_SIZE)
    video_valset = utils.create_video_dataset(valloader, NUM_FRAMES, OVERLAP_SIZE)
    video_testset = utils.create_video_dataset(testloader, NUM_FRAMES, OVERLAP_SIZE)

    # create the directory if it doesn't exist
    if not os.path.exists('./past_informed_videos'):
        os.makedirs('./past_informed_videos')

    # save in disk
    torch.save(video_trainset, './past_informed_videos/video_trainset.pt')
    torch.save(video_valset, './past_informed_videos/video_valset.pt')
    torch.save(video_testset, './past_informed_videos/video_testset.pt')


video_trainloader = DataLoader(video_trainset, batch_size=BATCH_SIZE, shuffle=False)
video_valloader = DataLoader(video_valset, batch_size=BATCH_SIZE, shuffle=False)
video_testloader = DataLoader(video_testset, batch_size=BATCH_SIZE, shuffle=False)


#_________________________DEVICE_____________________________
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('---------------------')
print('Using device:', device)
print('---------------------')

#_________________________TRAIN AND VALIDATION___________________________

criterion = nn.L1Loss()
criterion2 = nn.MSELoss()

param_grid = {'depth': [1, 2, 4, 8],
              'heads': [1, 2, 4, 8]}

best_val_loss = 1000

for depth in param_grid['depth']:
    for heads in param_grid['heads']:
        print('---------------------')
        print('Training with depth {} and heads {}'.format(depth, heads))
        print('---------------------')
        model = ViViT(image_size=IMAGE_SIZE, # according to the coordinates of interest 
            patch_size=PATCH_SIZE, 
            num_frames=NUM_FRAMES,
            in_channels=CHANNELS,   # according to the channels chosen
            dim=DIM, 
            depth=depth, 
            heads=heads).to(device)
    
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
            for inputs, labels in video_trainloader:
                inputs, labels = inputs.to(device), labels.float().to(device)

                optimizer.zero_grad()
                
                output = model(inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                if torch.isnan(loss):
                    print('Loss is nan. Stopping training.')
                    break

            train_loss /= (len(video_trainloader))
            train_losses.append(train_loss)

            # Evaluation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, labels in video_valloader:
                    inputs, labels = inputs.to(device), labels.float().to(device)
                    
                    output = model(inputs)
                    loss = criterion(output, labels)
                    val_loss += loss.item()
            
                val_loss /= (len(video_valloader))

                # if eval loss is lower than the previous one, save the hyperparameters
                if len(val_losses) == 0 or val_loss < min(val_losses):
                    no_improvement = 0
                else:
                    no_improvement +=1

                if no_improvement >= patience:
                    print('---------------------')
                    print('Early stopping in epoch {}'.format(epoch+1))
                    print('Current val loss: {:.4f}'.format(val_loss))
                    print('Optimal val loss: {:.4f}'.format(min(val_losses)))
                    print('Last patience val losses: ', np.round(val_losses[-patience], 4))
                    print('---------------------')
                          
                    break

                val_losses.append(val_loss)

        print('Train and val losses for depth {} and heads {}:'.format(depth, heads))
        print('Train loss: {:.4f}'.format(np.round(train_losses[-patience], 4)))
        print('Val loss: {:.4f}'.format(np.round(val_losses[-patience], 4)))
        print('---------------------')  

        if val_losses[-patience] < best_val_loss:
            best_val_loss = val_losses[-patience]
            best_val_losses = val_losses
            best_train_losses = train_losses
            best_depth = depth
            best_heads = heads
            best_epoch = epoch-patience

DEPTH = best_depth
HEADS = best_heads
EPOCHS = int(best_epoch*1.5)

print('Best depth and heads: {}, {}'.format(best_depth, best_heads))
print('---------------------')
print('Starting training with best hyperparameters...')
print('---------------------')

# plot losses
plot = input("Do you want to plot the losses? (y/n)")
if plot == "y":
    plt.plot(best_train_losses, label='Training Loss')
    plt.plot(best_val_losses, label='Validation Loss')
    plt.legend()
    # save figure in a document with the name of the model and the date
    plt.savefig('./figures/LOSSES_past_label_informed_ViViT_NHOURS{}_img{}_ptch{}_dpth{}_hds{}_{}.png'.format(N_HOURS,
                                                                                IMAGE_SIZE,
                                                                                PATCH_SIZE,
                                                                                DEPTH,
                                                                                HEADS,
                                                                                date_string))
    

# merge train and validation sets
video_trainset = torch.utils.data.ConcatDataset([video_trainset, video_valset])
video_trainloader = DataLoader(video_trainset, batch_size=BATCH_SIZE, shuffle=False)


#_________________________TRAINING THE MODEL___________________________


#__________________________MODEL_____________________________
model = ViViT(image_size=IMAGE_SIZE, # according to the coordinates of interest 
            patch_size=PATCH_SIZE, 
            num_frames=NUM_FRAMES,
            in_channels=CHANNELS,   # according to the channels chosen
            dim=DIM, 
            depth=DEPTH, 
            heads=HEADS).to(device)

parameters = filter(lambda p: p.requires_grad, model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) 
print('---------------------')
print('Trainable Parameters:', parameters)
print('---------------------')

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training
model.train()

for epoch in range(EPOCHS):
    train_loss = 0
    for inputs, labels in video_trainloader:
        inputs, labels = inputs.to(device), labels.float().to(device)

        optimizer.zero_grad()
        
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if torch.isnan(loss):
            print('Loss is nan. Stopping training.')
            break

    train_loss /= (len(video_trainloader))

    print(f'EPOCH {epoch+1}/{EPOCHS}, TRAIN_NMAE={train_loss:.4f}')


#_________________________SAVE MODEL___________________________

# save trained model
torch.save(model.state_dict(), './models/{}_nhours_past_label_informed_vivit.pt'.format(N_HOURS))

#_________________________EVALUATION___________________________


# Evaluation
model.eval()
nmae = 0
nmse = 0
with torch.no_grad():
    for inputs, labels in video_testloader:
        inputs, labels = inputs.to(device), labels.float().to(device)
        output = model(inputs)
        nmae += criterion(output, labels)
        nmse += criterion2(output, labels)

    nmae /= (len(video_testloader))
    nmse /= (len(video_testloader))


print('-------------------')
print('NMAE: ', nmae)
print('NMSE: ', nmse)



ans = input('Do you want to save results and characteristics of the model? (y/n)')
if ans=='y':
    # write evaluation results to file
    with open('./results/{}_NHOURS{}_img{}_ptch{}_dpth{}_hds{}_{}.txt'.format('past_label_informed_vivit',
                                            N_HOURS,                                  
                                            IMAGE_SIZE,
                                            PATCH_SIZE,
                                            DEPTH,
                                            HEADS,
                                            date_string), 'w') as f:
                
        f.write('TRAINING DATASET: \n')
        f.write(f'Number of samples: {len(video_trainset)} \n')
                
        f.write('TRAINING PARAMETERS: \n')
        f.write(f'Batch size: {BATCH_SIZE} Learning rate: {LEARNING_RATE} Epochs: {EPOCHS} \n')
        f.write(f'Trainable parameters:{parameters} \n')

        f.write('HYPERPARAMETERS: \n')
        f.write(f'Lat and Lon {MAX_LAT},{MIN_LAT}; {MAX_LON},{MIN_LON} \n')
        f.write(f'Image size: {IMAGE_SIZE} \n Patch size: {PATCH_SIZE} \n Channels: {CHANNELS} \n Dim: {DIM} \n Depth: {DEPTH} \n Heads: {HEADS} \n')
        f.write('\n')
        f.write('EVALUATION RESULTS: \n')
        f.write(f'NMAE: {nmae:.4f} \n')
        f.write(f'NMSE: {nmse:.4f} \n')

