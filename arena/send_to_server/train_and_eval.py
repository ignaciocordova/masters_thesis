import pandas as pd
import numpy as np

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import utils
from my_models import regression_ViT as ViT

import warnings
warnings.filterwarnings("ignore", 'This pattern has match groups')


#_________________DATA LOADED INTO MEMORY_____________________________
#PATH = './data/'

#training_df = pd.read_csv(PATH+'EjemploDatos.csv')
#target_df = pd.read_csv(PATH+'EjemploTarget.csv', header=None)

#test_df = pd.read_csv(PATH+'EjemploDatos.csv')
#test_target_df = pd.read_csv(PATH+'EjemploTarget.csv', header=None)

#_________TRAINING DATA_____________
#data, labels = utils.get_data_and_target(training_df, target_df, coordenates_of_interest, channels, normalize_target=False)
#trainset = torch.utils.data.TensorDataset(data, labels)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

#_________TEST DATA_________________
#test_data, test_labels = utils.get_data_and_target(test_df, test_target_df, coordenates_of_interest, channels, normalize_target=False)
#testset = torch.utils.data.TensorDataset(test_data, test_labels)
#testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)


#_________________________PARAMETERS___________________________
BATCH_SIZE = 2
EPOCHS = 2
LEARNING_RATE = 0.001
INSTALLED_POWER = 17500

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



#_________________DATA ACCESED FROM DISK_____________________________


df_file_path = './data/EjemploDatos.csv'
target_file_path = './data/EjemploTarget.csv'
class CustomDataset(Dataset):
    def __init__(self, df_file_path, target_file_path, coordinates_of_interest, channels, normalize_target=True,
                 INSTALLED_POWER=INSTALLED_POWER, image_size=9):
        self.df_file_path = df_file_path
        self.target_file_path = target_file_path
        self.coordinates_of_interest = coordinates_of_interest
        self.channels = channels
        self.normalize_target = normalize_target
        self.INSTALLED_POWER = INSTALLED_POWER
        self.image_size = image_size

        # read target file to memory
        self.target_df = pd.read_csv(target_file_path)

    def __len__(self):
        return len(self.target_df)

    def __getitem__(self, idx):
        # read df file for specific row
        row_df = pd.read_csv(self.df_file_path, skiprows=range(1, idx + 1), nrows=1)

        # keep columns that contain any of the coordinates of interest
        row_df = row_df.filter(regex='|'.join(self.coordinates_of_interest))

        # get data of interest
        image = self.create_image(row_df, self.channels, image_size=self.image_size)

        # get target of interest
        target = torch.tensor(self.target_df.iloc[idx, 1], dtype=torch.float32)

        if self.normalize_target:
            target = target / self.INSTALLED_POWER

        return image, target

    def create_image(self, df, channels, image_size):
        """
        Creates images from the data in df.

        Parameters
        ----------
        df : pandas dataframe
            Dataframe with all the data.
        channels : list
            List with the channel names to build the image.
        image_size : int
            Size of the image.

        Returns
        -------
        images : torch tensor
            Shape:  (n_channels, image_size, image_size)
            Tensor with all the images of the coordinates of interest.

        """
        images = np.zeros((len(channels), image_size, image_size), dtype=np.float32)

        for i, channel in enumerate(channels):
            channel_data = df.filter(regex=channel).values.reshape(image_size, image_size)
            images[i, :, :] = channel_data

        return torch.from_numpy(images)

trainset = CustomDataset(df_file_path, target_file_path, coordinates_of_interest, channels)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


#__________________________MODEL_____________________________
model = ViT(image_size=9, # according to the coordinates of interest 
            patch_size=3, 
            channels=8,   # according to the channels chosen
            dim=64, 
            depth=6, 
            heads=8, 
            mlp_dim=128)


#_________________________TRAINING THE MODEL___________________________
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

n_total_steps = len(trainloader)

# train the network
for epoch in range(EPOCHS):  # loop over the dataset multiple times

    print('Starting epoch: ', epoch+1, '/', EPOCHS, '...')

    for i, data in enumerate(trainloader, 0):

        print(i)
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        print('')
        print('inputs.shape: ', inputs.shape)
        print('labels.shape: ', labels.shape)
        print('')

        labels = labels.unsqueeze(1).float()

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

#_________________________TESTING THE MODEL___________________________
total_samples = len(testloader.dataset)
total_loss = 0
total_loss2 = 0

criterion2 = nn.L1Loss()

print('')
print('Testing the model...')
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
