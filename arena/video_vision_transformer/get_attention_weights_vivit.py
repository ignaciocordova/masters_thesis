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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

print('Loading model 2')
#load model
model = ViViT(image_size=IMAGE_SIZE, # according to the coordinates of interest 
            patch_size=PATCH_SIZE, 
            num_frames=4,
            in_channels=CHANNELS,   # according to the channels chosen
            dim=DIM, 
            depth=4, 
            heads=1).to(device)
print('Loading state dictironary for model 2')
model.load_state_dict(torch.load('./models/vivit.pt'))


testset = torch.load('./videos/video_testset.pt')
testloader = DataLoader(testset, batch_size=1, shuffle=False)

from recorder import Recorder
v = Recorder(model)


attention_scores_cls = []

model.eval()
predictions2 = []
for i, (data, labels) in enumerate(testloader):
    data = data.to(device)
    preds, attns = v(data)
    attention_scores_cls.append(attns[0][:, :, 0, :].cpu().detach().numpy())  # batch x depth x head x token


# Convert the list to a NumPy array for further manipulation
attention_scores_cls = np.array(attention_scores_cls)

# Calculate the mean along axis 0 to average attention scores
average_attention_scores_cls = np.mean(attention_scores_cls, axis=0)


average_attention_scores_cls=average_attention_scores_cls.reshape(4,5)

fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

# Set common x-axis labels
x_ticks_labels = ['rgs token', 'T-4', 'T-3', 'T-2', 'T-1']

# Set y-axis limits
y_axis_limits = (0, 1.0)

# Customize each subplot
for i in range(4):
    axs[i].bar(np.arange(5), average_attention_scores_cls[i], color='peru', edgecolor='black')
    axs[i].set_ylim(y_axis_limits)

# Set common x-axis properties
axs[-1].set_xticks(np.arange(5))
axs[-1].set_xticklabels(x_ticks_labels)
axs[-1].set_xlabel('Regression Token & Temporal Tokens (video frames)')

plt.tight_layout()
plt.savefig('./figures/temporal_attention_weights.png')
