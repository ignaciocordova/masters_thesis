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

NUM_FRAMES = 8 # number of frames in the video
OVERLAP_SIZE = 7 #number of overlaping frames

BATCH_SIZE = 64
EPOCHS = 150
LEARNING_RATE = 0.0001

IMAGE_SIZE = 9 
PATCH_SIZE = 3
CHANNELS = 8
DIM = 64
DEPTH = 2       # number of transformer blocks
HEADS = 2

#_________________________DATA OF INTEREST_____________________________
coordinates_of_interest = ['prediction date']

for lat in np.arange(MAX_LAT, MIN_LAT-0.125, -0.125):
    for lon in np.arange(MIN_LON, MAX_LON+0.125, 0.125):
        coordinates_of_interest.append(f'({lat}, {lon})')

channels = ['10u', '10v', '2t', 'sp', '100u', '100v', 'vel10_', 'vel100']


#______________DATA________________

# ask the user if he wants to create the train and test sets or load them from disk
answer = input('Do you want to create the IMAGE train and test sets? (y/n)')

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
    # load the train and test sets from disk
    trainset = torch.load('./processed_data/trainset.pt')
    testset = torch.load('./processed_data/testset.pt')

print('Size of images trainset: {}'.format(len(trainset)))
print('Size of images testset: {}'.format(len(testset)))
print('')

trainloader = DataLoader(trainset, batch_size=1, shuffle=False)
testloader = DataLoader(testset, batch_size=1, shuffle=False)

# OVERLAPPING VIDEO DATA
# ask the user if he wants to create the train and test sets or load them from disk
answer = input('Do you want to create the overlap VIDEO train and test sets? (y/n)')
if answer == 'y':
    # iterate over the trainloader and create the video trainset
    video_trainset = []
    video = []
    for i, (image, label) in enumerate(trainloader):
        video.extend(image)
        if len(video) == NUM_FRAMES:
            for j in range(NUM_FRAMES - OVERLAP_SIZE):
                video_trainset.append((torch.stack(video[j:j+NUM_FRAMES]), label))
            video = video[NUM_FRAMES - OVERLAP_SIZE:]

    # handle the remaining frames that don't form a complete video
    if len(video) > 0:
        for j in range(len(video) - NUM_FRAMES + 1):
            video_trainset.append((torch.stack(video[j:j+NUM_FRAMES]), label))


    # iterate over the testloader and create the video testset
    video_testset = []
    video = []
    for i, (image, label) in enumerate(testloader):
        video.extend(image)
        if len(video) == NUM_FRAMES:
            for j in range(NUM_FRAMES - OVERLAP_SIZE):
                video_testset.append((torch.stack(video[j:j+NUM_FRAMES]), label))
            video = video[NUM_FRAMES - OVERLAP_SIZE:]
    
    # handle the remaining frames that don't form a complete video
    if len(video) > 0:
        for j in range(len(video) - NUM_FRAMES + 1):
            video_testset.append((torch.stack(video[j:j+NUM_FRAMES]), label))

    
    # create the directory if it doesn't exist
    if not os.path.exists('./overlap_processed_data'):
        os.makedirs('./overlap_processed_data')

    # save in disk
    torch.save(video_trainset, './overlap_processed_data/video_trainset.pt')
    torch.save(video_testset, './overlap_processed_data/video_testset.pt')

else:
    # load the train and test sets from disk
    video_trainset = torch.load('./overlap_processed_data/video_trainset.pt')
    video_testset = torch.load('./overlap_processed_data/video_testset.pt')

video_trainloader = DataLoader(video_trainset, batch_size=BATCH_SIZE, shuffle=False)
video_testloader = DataLoader(video_testset, batch_size=BATCH_SIZE, shuffle=False)

# print a complete analysis of the dimension of data and labels of video_trainset and video_testset
print('First label should be 9886.56: {}'.format(video_trainset[0][1]))

# print a complete analysis of trainset shapes and sizes 
print('Trainset shapes and sizes:')
print('I expect it to be input [32 8 8 9 9] and label [32 1]')
for i, (image, label) in enumerate(video_trainloader):

    print('Input shape: {}'.format(image.shape))
    print('Label shape: {}'.format(label.shape))
    print('')
    print('Input size: {}'.format(image.size()))
    print('Label size: {}'.format(label.size()))
    print('')
    print('Input type: {}'.format(image.type()))
    print('Label type: {}'.format(label.type()))
    print('')
    print('Input dtype: {}'.format(image.dtype))
    print('Label dtype: {}'.format(label.dtype))
    print('')
    print('Input ndim: {}'.format(image.ndim))
    print('Label ndim: {}'.format(label.ndim))
    print('')
    break 

print('Number of training samples: {}'.format(len(video_trainset)))
print('Number of testing samples: {}'.format(len(video_testset)))


#_________________________DEVICE_____________________________
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#__________________________MODEL_____________________________
model = ViViT(image_size=IMAGE_SIZE, # according to the coordinates of interest 
            patch_size=PATCH_SIZE, 
            num_frames=NUM_FRAMES,
            in_channels=CHANNELS,   # according to the channels chosen
            dim=DIM, 
            depth=DEPTH, 
            heads=HEADS).to(device)

#_________________________TRAINING AND TESTING THE MODEL___________________________

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
eval_losses = []

n_batches = len(video_testloader)

# train and evaluate the network
for epoch in range(EPOCHS):

    # Training
    model.train()
    train_loss = 0
    for i, (inputs, labels) in enumerate(video_trainloader):
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

    train_loss /= (len(video_trainloader)*INSTALLED_POWER)
    train_losses.append(train_loss)         

    # Evaluation
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for inputs, labels in video_testloader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            output = model(inputs)
            eval_loss += criterion(output, labels).item()/INSTALLED_POWER

    eval_loss /= n_batches
    eval_losses.append(eval_loss)

    print(f'EPOCH {epoch+1}/{EPOCHS}, TRAIN_LOSS={train_loss:.4f}, EVAL_LOSS={eval_loss:.4f}')

date_string = datetime.now().strftime("%m_%d-%I_%M_%p")

print('Finished Training')

# plot losses
plot_opt = input("Do you want to plot the losses? (y/n)")
if plot_opt == "y":
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
    for data, target in video_testloader:

        data = data.to(device)
        target = target.to(device)
        
        output = model(data)

        loss = criterion(output, target.float())
        total_loss += loss.item()/INSTALLED_POWER

        loss2 = criterion2(output, target.float())
        total_loss2 += loss2.item()/(INSTALLED_POWER**2)


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
                
        f.write('TRAINING PARAMETERS: \n')
        f.write(f'Batch size: {BATCH_SIZE} Learning rate: {LEARNING_RATE} Epochs: {EPOCHS} \n')

        f.write('HYPERPARAMETERS: \n')
        f.write(f'Lat and Lon {MAX_LAT},{MIN_LAT}; {MAX_LON},{MIN_LON} \n')
        f.write(f'Image size: {IMAGE_SIZE} \n Patch size: {PATCH_SIZE} \n Channels: {CHANNELS} \n Dim: {DIM} \n Depth: {DEPTH} \n Heads: {HEADS} \n')
        f.write('\n')
        f.write('EVALUATION RESULTS: \n')
        f.write(f'NMAE: {nmae:.4f} \n')
        f.write(f'NMSE: {nmse:.4f} \n')


