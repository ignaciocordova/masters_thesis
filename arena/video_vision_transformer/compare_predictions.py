import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

import torch
from torch.utils.data import DataLoader

INSTALLED_POWER = 17500

target = pd.read_csv('target_stv_2018.csv', header=None, names=['target'])

# create displaced target 
target['target_displaced'] = target['target'].shift(1).fillna(target['target'].mean())

dumb_predictions = target['target_displaced'].values

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

#load model 
model1 = torch.load('./models/vivit.pt')

testset = torch.load('./videos/video_testset.pt')
testloader = DataLoader(testset, batch_size=1, shuffle=False)

with torch.no_grad():
    model1.eval()
    predictions1 = []
    for i, (data, labels) in enumerate(testloader):
        data = data.to(device)
        labels = labels.to(device)
        output = model1(data)
        predictions1.append(output.item()*INSTALLED_POWER)

#load model
model2 = torch.load('./models/past_label_informed_vivit.pt')

testset = torch.load('./past_informed_videos/video_testset.pt')
testloader = DataLoader(testset, batch_size=1, shuffle=False)

with torch.no_grad():
    model2.eval()
    predictions2 = []
    for i, (data, labels) in enumerate(testloader):
        data = data.to(device)
        labels = labels.to(device)
        output = model2(data)
        predictions2.append(output.item()*INSTALLED_POWER)

# plot predictions vs labels
plt.figure(figsize=(20, 10))
plt.plot(target['target'].values[::10], label='labels',color='black',linewidth=2)
plt.plot(predictions1[::10], label='ViViT')
plt.plot(predictions2[::10], label='Past informed ViViT',linewidth=1.3)
plt.plot(dumb_predictions[::10], label='dumb_predictions', color='gray')
plt.legend()
plt.ylabel('Power (kW)')
plt.xlabel('Time (h)')
plt.savefig('./figures/predictions_vs_labels.png')

# plot predictions vs labels zoomed
plt.figure(figsize=(20, 10))
plt.plot(target['target'].values[950:1000], label='labels',color='black',linewidth=2)
plt.plot(predictions1[950:1000], label='ViViT')
plt.plot(predictions2[950:1000], label='Past informed ViViT',linewidth=1.3)
plt.plot(dumb_predictions[950:1000], label='dumb_predictions', color='gray')
plt.legend()
plt.ylabel('Power (kW)')
plt.xlabel('Time (h)')
plt.savefig('./figures/predictions_vs_labels_zoomed.png')

# plot predictions vs labels zoomed,1.3
plt.figure(figsize=(20, 10))
plt.plot(target['target'].values[10:60], label='labels',color='black',linewidth=2)
plt.plot(predictions1[10:60], label='ViViT')
plt.plot(predictions2[10:60], label='Past informed ViViT',linewidth=1.3)
plt.plot(dumb_predictions[10:60], label='dumb_predictions', color='gray')
plt.legend()
plt.ylabel('Power (kW)')
plt.xlabel('Time (h)')
plt.savefig('./figures/predictions_vs_labels_zoomed2.png')
