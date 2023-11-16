import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
target['target_displaced'] = target['target'].shift(1).fillna(target['target'].mean())
dumb_predictions = target['target_displaced'].values

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

#load model
print('Loading model 1')
model1 = ViT(image_size=IMAGE_SIZE, # according to the coordinates of interest 
            patch_size=PATCH_SIZE, 
            channels=CHANNELS,   # according to the channels chosen
            dim=DIM, 
            depth=4, 
            heads=8, 
            mlp_dim=MLP_DIM).to(device)

print('Loading state dictironary for model 1')
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

print('Loading model 2')
#load model
model2 = ViViT(image_size=IMAGE_SIZE, # according to the coordinates of interest 
            patch_size=PATCH_SIZE, 
            num_frames=4,
            in_channels=CHANNELS,   # according to the channels chosen
            dim=DIM, 
            depth=4, 
            heads=1).to(device)
print('Loading state dictironary for model 2')
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
print('Loading model 3')
model3 = ViT(image_size=IMAGE_SIZE, # according to the coordinates of interest 
            patch_size=PATCH_SIZE, 
            channels=CHANNELS,   # according to the channels chosen
            dim=DIM, 
            depth=8, 
            heads=2, 
            mlp_dim=MLP_DIM).to(device)
print('Loading state dictironary for model 3')
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
print('Loading model 4')
model4 = ViViT(image_size=IMAGE_SIZE, # according to the coordinates of interest 
            patch_size=PATCH_SIZE, 
            num_frames=4,
            in_channels=CHANNELS,   # according to the channels chosen
            dim=DIM, 
            depth=4, 
            heads=4).to(device)
print('Loading state dictironary for model 4')
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


# Normalized Mean Absolute Error (NMAE) and Normalized Mean Squared Error (NMSE) calculation function
def calculate_nmae_nmse(predictions, actuals, installed_power):
    nmae = mean_absolute_error(actuals, predictions) / installed_power
    nmse = mean_squared_error(actuals, predictions) / installed_power**2
    return nmae, nmse

# Actual target values
actuals = target['target'].values

# Calculate NMAE and NMSE for each model
nmae1, nmse1 = calculate_nmae_nmse(predictions1, actuals, INSTALLED_POWER)
nmae2, nmse2 = calculate_nmae_nmse(predictions2, actuals, INSTALLED_POWER)
nmae3, nmse3 = calculate_nmae_nmse(predictions3, actuals, INSTALLED_POWER)
nmae4, nmse4 = calculate_nmae_nmse(predictions4, actuals, INSTALLED_POWER)

# Print NMAE and NMSE for each model
print('NMAE and NMSE for ViT: ', nmae1, nmse1)
print('NMAE and NMSE for ViViT: ', nmae2, nmse2)
print('NMAE and NMSE for AR-ViT: ', nmae3, nmse3)
print('NMAE and NMSE for AR-ViViT: ', nmae4, nmse4)

predictions1 = predictions1[1:]
predictions2 = predictions2[1:]
predictions3 = predictions3[1:]
predictions4 = predictions4[1:]


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def enhance_plot(ax, title, x_label, y_label, legend_fontsize=14):  # Added legend_fontsize parameter
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(loc='best', fontsize=legend_fontsize, frameon=True, shadow=True)  # Using the legend_fontsize parameter
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

# Plot predictions vs labels
plt.figure(figsize=(25, 10))
plt.plot(target['target'].values[5000:7000:10], label='Observed', color='black', linewidth=2, linestyle='-', marker='o')
plt.plot(predictions1[5000:7000:10], label='ViT', linewidth=1.3, linestyle='--', marker='x')
plt.plot(predictions2[5000:7000:10], label='ViViT', linewidth=1.3, linestyle='-.', marker='+')
plt.plot(predictions3[5000:7000:10], label='AR-ViT', linewidth=1.3, linestyle=':', marker='^')
plt.plot(predictions4[5000:7000:10], label='AR-ViViT', linewidth=1.3, linestyle='--', marker='v')
enhance_plot(plt.gca(), 'Wind Power Forecasting: Predictions vs Observed', 'Time (every 10 hours, up to 2000 hours)', 'Power (kW)')
plt.savefig('./figures/predictions_vs_labels.png', dpi=300)


# Plot predictions vs labels zoomed
plt.figure(figsize=(20, 10))
plt.plot(target['target'].values[800:860], label='Observed', color='black', linewidth=2, linestyle='-', marker='o')
plt.plot(predictions1[800:860], label='ViT', linewidth=1.5, linestyle='--', marker='x')
plt.plot(predictions2[800:860], label='ViViT', linewidth=1.5, linestyle='-.', marker='+')
plt.plot(predictions3[800:860], label='AR-ViT', linewidth=1.5, linestyle=':', marker='^')
plt.plot(predictions4[800:860], label='AR-ViViT', linewidth=1.5, linestyle='--', marker='v')
enhance_plot(plt.gca(), 'Wind Power Forecasting: Predictions vs Observed (Zoomed View 1)', 'Time (h)', 'Power (kW)')
plt.savefig('./figures/predictions_vs_labels_zoomed.png', dpi=300)

# Plot predictions vs labels zoomed, 1.3
plt.figure(figsize=(20, 10))
plt.plot(target['target'].values[60:120], label='Observed', color='black', linewidth=2, linestyle='-', marker='o')
plt.plot(predictions1[60:120], label='ViT', linewidth=1.5, linestyle='--', marker='x')
plt.plot(predictions2[60:120], label='ViViT', linewidth=1.5, linestyle='-.', marker='+')
plt.plot(predictions3[60:120], label='AR-ViT', linewidth=1.5, linestyle=':', marker='^')
plt.plot(predictions4[60:120], label='AR-ViViT', linewidth=1.5, linestyle='--', marker='v')
enhance_plot(plt.gca(), 'Wind Power Forecasting: Predictions vs Observed (Zoomed View 2)', 'Time (h)', 'Power (kW)')
plt.savefig('./figures/predictions_vs_labels_zoomed2.png', dpi=300)

# Plot predictions vs labels zoomed, 1.3
plt.figure(figsize=(20, 10))
plt.plot(target['target'].values[5800:5860], label='Observed', color='black', linewidth=2, linestyle='-', marker='o')
plt.plot(predictions1[5800:5860], label='ViT', linewidth=1.5, linestyle='--', marker='x')
plt.plot(predictions2[5800:5860], label='ViViT', linewidth=1.5, linestyle='-.', marker='+')
plt.plot(predictions3[5800:5860], label='AR-ViT', linewidth=1.5, linestyle=':', marker='^')
plt.plot(predictions4[5800:5860], label='AR-ViViT', linewidth=1.5, linestyle='--', marker='v')
enhance_plot(plt.gca(), 'Wind Power Forecasting: Predictions vs Observed (Zoomed View 2)', 'Time (h)', 'Power (kW)')
plt.savefig('./figures/predictions_vs_labels_zoomed3.png', dpi=300)

# Plot predictions vs labels zoomed, 1.3
plt.figure(figsize=(20, 10))
plt.plot(target['target'].values[8300:8360], label='Observed', color='black', linewidth=2, linestyle='-', marker='o')
plt.plot(predictions1[8300:8360], label='ViT', linewidth=1.5, linestyle='--', marker='x')
plt.plot(predictions2[8300:8360], label='ViViT', linewidth=1.5, linestyle='-.', marker='+')
plt.plot(predictions3[8300:8360], label='AR-ViT', linewidth=1.5, linestyle=':', marker='^')
plt.plot(predictions4[8300:8360], label='AR-ViViT', linewidth=1.5, linestyle='--', marker='v')
enhance_plot(plt.gca(), 'Wind Power Forecasting: Predictions vs Observed (Zoomed View 2)', 'Time (h)', 'Power (kW)')
plt.savefig('./figures/predictions_vs_labels_zoomed4.png', dpi=300)

# Plot predictions vs labels zoomed, 1.3
plt.figure(figsize=(20, 10))
plt.plot(target['target'].values[4100:4160], label='Observed', color='black', linewidth=2, linestyle='-', marker='o')
plt.plot(predictions1[4100:4160], label='ViT', linewidth=1.5, linestyle='--', marker='x')
plt.plot(predictions2[4100:4160], label='ViViT', linewidth=1.5, linestyle='-.', marker='+')
plt.plot(predictions3[4100:4160], label='AR-ViT', linewidth=1.5, linestyle=':', marker='^')
plt.plot(predictions4[4100:4160], label='AR-ViViT', linewidth=1.5, linestyle='--', marker='v')
enhance_plot(plt.gca(), 'Wind Power Forecasting: Predictions vs Observed (Zoomed View 2)', 'Time (h)', 'Power (kW)')
plt.savefig('./figures/predictions_vs_labels_zoomed5.png', dpi=300)

# Plot predictions vs labels zoomed, 1.3
plt.figure(figsize=(20, 10))
plt.plot(target['target'].values[2760:2820], label='Observed', color='black', linewidth=2, linestyle='-', marker='o')
plt.plot(predictions1[2760:2820], label='ViT', linewidth=1.5, linestyle='--', marker='x')
plt.plot(predictions2[2760:2820], label='ViViT', linewidth=1.5, linestyle='-.', marker='+')
plt.plot(predictions3[2760:2820], label='AR-ViT', linewidth=1.5, linestyle=':', marker='^')
plt.plot(predictions4[2760:2820], label='AR-ViViT', linewidth=1.5, linestyle='--', marker='v')
enhance_plot(plt.gca(), 'Wind Power Forecasting: Predictions vs Observed (Zoomed View 2)', 'Time (h)', 'Power (kW)')
plt.savefig('./figures/predictions_vs_labels_zoomed6.png', dpi=300)


# # plot predictions vs labels
# plt.figure(figsize=(25, 10))
# plt.plot(target['target'].values[::10], label='labels',color='black',linewidth=2)
# plt.plot(predictions1[::10], label='ViT', linewidth=1.3)
# plt.plot(predictions2[::10], label='ViViT',linewidth=1.3)
# plt.plot(predictions3[::10], label='ViT (past label informed)', linewidth=1.3)
# plt.plot(predictions4[::10], label='ViViT (past label informed)',linewidth=1.3)
# plt.legend()
# plt.ylabel('Power (kW)')
# plt.xlabel('Time (h)')
# plt.savefig('./figures/predictions_vs_labels.png')

# # plot predictions vs labels zoomed
# plt.figure(figsize=(20, 10))
# plt.plot(target['target'].values[950:1000], label='labels',color='black',linewidth=2)
# plt.plot(predictions1[950:1000], label='ViT', linewidth=1.3)
# plt.plot(predictions2[950:1000], label='ViViT',linewidth=1.3)
# plt.plot(predictions3[950:1000], label='ViT (past label informed)', linewidth=1.3)
# plt.plot(predictions4[950:1000], label='ViViT (past label informed)',linewidth=1.3)
# plt.legend()
# plt.ylabel('Power (kW)')
# plt.xlabel('Time (h)')
# plt.savefig('./figures/predictions_vs_labels_zoomed.png')

# # plot predictions vs labels zoomed,1.3
# plt.figure(figsize=(20, 10))
# plt.plot(target['target'].values[10:60], label='labels',color='black',linewidth=2)
# plt.plot(predictions1[10:60], label='ViT', linewidth=1.3)
# plt.plot(predictions2[10:60], label='ViViT',linewidth=1.3)
# plt.plot(predictions3[10:60], label='ViT (past label informed)', linewidth=1.3)
# plt.plot(predictions4[10:60], label='ViViT (past label informed)',linewidth=1.3)
# plt.legend() 
# plt.ylabel('Power (kW)')
# plt.xlabel('Time (h)') 
# plt.savefig('./figures/predictions_vs_labels_zoomed2.png')
