import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

import torch
from torch.utils.data import DataLoader

from my_models import regression_ViT as ViT, ViViT

INSTALLED_POWER = 17500
IMAGE_SIZE = 9 
PATCH_SIZE = 3
CHANNELS = 8
DIM = 64
MLP_DIM = 64  

target = pd.read_csv('target_stv_2018.csv', header=None, names=['target'])

# create displaced target 
# target['target_displaced'] = target['target'].shift(1).fillna(target['target'].mean())
# dumb_predictions = target['target_displaced'].values

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

#load model 
model1 = ViT(image_size=IMAGE_SIZE, # according to the coordinates of interest 
            patch_size=PATCH_SIZE, 
            channels=CHANNELS,   # according to the channels chosen
            dim=DIM, 
            depth=4, 
            heads=4, 
            mlp_dim=MLP_DIM).to(device)
model1.load_state_dict(torch.load('./models/vit.pt'))

parameters = filter(lambda p: p.requires_grad, model1.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) 
print('---------------------')
print('Trainable Parameters:', parameters)
print('---------------------')

testset = torch.load('./images/testset.pt')
testloader = DataLoader(testset, batch_size=1, shuffle=False)

model1.eval()
predictions1 = []
for i, (data, labels) in enumerate(testloader):
    data = data.to(device)
    labels = labels.to(device)
    output = model1(data)
    predictions1.append(output.item()*INSTALLED_POWER)

#load model
model2 = ViViT(image_size=IMAGE_SIZE, # according to the coordinates of interest 
            patch_size=PATCH_SIZE, 
            num_frames=4,
            in_channels=CHANNELS,   # according to the channels chosen
            dim=DIM, 
            depth=4, 
            heads=1).to(device)
model2.load_state_dict(torch.load('./models/vivit.pt'))


testset = torch.load('./videos/video_testset.pt')
testloader = DataLoader(testset, batch_size=1, shuffle=False)

model2.eval()
predictions2 = []
for i, (data, labels) in enumerate(testloader):
    data = data.to(device)
    labels = labels.to(device)
    output = model2(data)
    predictions2.append(output.item()*INSTALLED_POWER)

CHANNELS = 8+1

# load model
model3 = ViT(image_size=IMAGE_SIZE, # according to the coordinates of interest 
            patch_size=PATCH_SIZE, 
            channels=CHANNELS,   # according to the channels chosen
            dim=DIM, 
            depth=8, 
            heads=2, 
            mlp_dim=MLP_DIM).to(device)
model3.load_state_dict(torch.load('./models/past_label_informed_vit.pt'))


testset = torch.load('./past_informed_images/testset.pt')
testloader = DataLoader(testset, batch_size=1, shuffle=False)

model3.eval()
predictions3 = []
for i, (data, labels) in enumerate(testloader):
    data = data.to(device)
    labels = labels.to(device)
    output = model3(data)
    predictions3.append(output.item()*INSTALLED_POWER)

# load model
model4 = ViViT(image_size=IMAGE_SIZE, # according to the coordinates of interest 
            patch_size=PATCH_SIZE, 
            num_frames=4,
            in_channels=CHANNELS,   # according to the channels chosen
            dim=DIM, 
            depth=4, 
            heads=4).to(device)
model4.load_state_dict(torch.load('./models/past_label_informed_vivit.pt'))

testset = torch.load('./past_informed_videos/video_testset.pt')
testloader = DataLoader(testset, batch_size=1, shuffle=False)

model4.eval()
predictions4 = []
for i, (data, labels) in enumerate(testloader):
    data = data.to(device)
    labels = labels.to(device)
    output = model4(data)
    predictions4.append(output.item()*INSTALLED_POWER)


# plot predictions vs labels
plt.figure(figsize=(25, 10))
plt.plot(target['target'].values[::10], label='labels',color='black',linewidth=2)
plt.plot(predictions1[::10], label='ViT', linewidth=1.3)
plt.plot(predictions2[::10], label='ViViT',linewidth=1.3)
plt.plot(predictions3[::10], label='ViT (past label informed)', linewidth=1.3)
plt.plot(predictions4[::10], label='ViViT (past label informed)',linewidth=1.3)
plt.legend()
plt.ylabel('Power (kW)')
plt.xlabel('Time (h)')
plt.savefig('./figures/predictions_vs_labels.png')

# plot predictions vs labels zoomed
plt.figure(figsize=(20, 10))
plt.plot(target['target'].values[950:1000], label='labels',color='black',linewidth=2)
plt.plot(predictions1[950:1000], label='ViT', linewidth=1.3)
plt.plot(predictions2[950:1000], label='ViViT',linewidth=1.3)
plt.plot(predictions3[950:1000], label='ViT (past label informed)', linewidth=1.3)
plt.plot(predictions4[950:1000], label='ViViT (past label informed)',linewidth=1.3)
plt.legend()
plt.ylabel('Power (kW)')
plt.xlabel('Time (h)')
plt.savefig('./figures/predictions_vs_labels_zoomed.png')

# plot predictions vs labels zoomed,1.3
plt.figure(figsize=(20, 10))
plt.plot(target['target'].values[10:60], label='labels',color='black',linewidth=2)
plt.plot(predictions1[10:60], label='ViT', linewidth=1.3)
plt.plot(predictions2[10:60], label='ViViT',linewidth=1.3)
plt.plot(predictions3[10:60], label='ViT (past label informed)', linewidth=1.3)
plt.plot(predictions4[10:60], label='ViViT (past label informed)',linewidth=1.3)
plt.legend()
plt.ylabel('Power (kW)')
plt.xlabel('Time (h)')
plt.savefig('./figures/predictions_vs_labels_zoomed2.png')
