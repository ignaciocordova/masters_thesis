import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler

import utils
from my_models import ViViT as overlap_vivit

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
EPOCHS = 150
LEARNING_RATE = 0.0001

IMAGE_SIZE = 9 
PATCH_SIZE = 3
CHANNELS = 8+1 # 8 channels + 1 previous label channel
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
    meta_data = pd.concat([pd.read_csv('data_stv_2016.csv'),
                            pd.read_csv('data_stv_2017.csv')],
                            axis=0)

    # feature scale the data ignoring the first column
    scaler = StandardScaler()
    meta_data.iloc[:, 1:] = scaler.fit_transform(meta_data.iloc[:, 1:])

    # TRAINING TARGET
    meta_target = pd.concat([pd.read_csv('target_stv_2016.csv', header=None),
                              pd.read_csv('target_stv_2017.csv', header=None)],
                              axis=0)

    # TEST DATA
    df_2018 = pd.read_csv('data_stv_2018.csv')

    # feature scale the data ignoring the first column
    df_2018.iloc[:, 1:] = scaler.transform(df_2018.iloc[:, 1:])

    # TEST TARGET
    target_2018 = pd.read_csv('target_stv_2018.csv', header=None)

    # warning if meta_data and meta_target have different number of rows
    if meta_data.shape[0] != meta_target.shape[0]:
        warnings.warn('meta_data has {} rows and meta_target has {} rows'.format(meta_data.shape[0], meta_target.shape[0]), RuntimeWarning)

    data, labels = utils.get_data_and_target_with_previous_label_channel(meta_data, meta_target, coordinates_of_interest, channels, IMAGE_SIZE, normalize_target=True)
    trainset = torch.utils.data.TensorDataset(data, labels)

    test_data, test_labels = utils.get_data_and_target_with_previous_label_channel(df_2018, target_2018, coordinates_of_interest, channels, IMAGE_SIZE, normalize_target=True)
    # first image in test set has label channel of last in train set
    test_data[0, -1, :, :] = trainset[-1][1]
    testset = torch.utils.data.TensorDataset(test_data, test_labels)

    # create the directory if it doesn't exist
    if not os.path.exists('./past_informed_images'):
        os.makedirs('./past_informed_images')
    
    # save in disk
    torch.save(trainset, './past_informed_images/trainset.pt')
    torch.save(testset, './past_informed_images/testset.pt')

else:
    # load the train and test sets from disk
    trainset = torch.load('./past_informed_images/trainset.pt')
    testset = torch.load('./past_informed_images/testset.pt')

print('Size of images trainset: {}'.format(len(trainset)))
print('Size of images testset: {}'.format(len(testset)))
print('')

trainloader = DataLoader(trainset, batch_size=1, shuffle=False)
testloader = DataLoader(testset, batch_size=1, shuffle=False)

# OVERLAPPING VIDEO DATA
# ask the user if he wants to create the train and test sets or load them from disk
answer = input('Do you want to create the overlap VIDEO train and test sets? (y/n)')
if answer == 'y':
    
    video_trainset = utils.create_video_dataset(trainloader, NUM_FRAMES, OVERLAP_SIZE)
    print('Size of video trainset: {}'.format(len(video_trainset)))
    video_testset = utils.create_video_dataset(testloader, NUM_FRAMES, OVERLAP_SIZE)

    # create the directory if it doesn't exist
    if not os.path.exists('./past_informed_videos'):
        os.makedirs('./past_informed_videos')

    # save in disk
    torch.save(video_trainset, './past_informed_videos/video_trainset.pt')
    torch.save(video_testset, './past_informed_videos/video_testset.pt')

else:
    # load the train and test sets from disk
    video_trainset = torch.load('./past_informed_videos/video_trainset.pt')
    video_testset = torch.load('./past_informed_videos/video_testset.pt')

video_trainloader = DataLoader(video_trainset, batch_size=BATCH_SIZE, shuffle=False)
print('Len of video_trainloader:', len(video_trainloader))
video_testloader = DataLoader(video_testset, batch_size=BATCH_SIZE, shuffle=False)

#_________________________DEVICE_____________________________
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#__________________________MODEL_____________________________
model = overlap_vivit(image_size=IMAGE_SIZE, # according to the coordinates of interest 
            patch_size=PATCH_SIZE, 
            num_frames=NUM_FRAMES,
            in_channels=CHANNELS,   # according to the channels chosen
            dim=DIM, 
            depth=DEPTH, 
            heads=HEADS).to(device)

parameters = filter(lambda p: p.requires_grad, model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) 
print('Trainable Parameters:', parameters)

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

    train_loss /= len(video_trainloader)
    train_losses.append(train_loss)         

    # Evaluation
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for inputs, labels in video_testloader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            output = model(inputs)
            eval_loss += criterion(output, labels).item()

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
        total_loss += loss.item()

        loss2 = criterion2(output, target.float())
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
    torch.save(model, './models/past_label_informed_vivit_2enc2heads.pt')



