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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

#load model
print('Loading model')
model = ViT(image_size=IMAGE_SIZE, # according to the coordinates of interest 
            patch_size=PATCH_SIZE, 
            channels=CHANNELS,   # according to the channels chosen
            dim=DIM, 
            depth=4, 
            heads=8, 
            mlp_dim=MLP_DIM).to(device)

model.load_state_dict(torch.load('./models/vit.pt'))

from recorder import Recorder
v = Recorder(model)

testset = torch.load('./images/testset.pt')
testloader = DataLoader(testset, batch_size=1, shuffle=False)

# Initialize an empty list to store attention scores
attention_scores_00 = []
attention_scores_01 = []
attention_scores_02 = []
attention_scores_03 = []

attention_scores_00 = []
attention_scores_01 = []
attention_scores_02 = []
attention_scores_03 = []

attention_scores_00 = []
attention_scores_01 = []
attention_scores_02 = []
attention_scores_03 = []

attention_scores_10 = []
attention_scores_11 = []
attention_scores_12 = []
attention_scores_13 = []

attention_scores_20 = []
attention_scores_21 = []
attention_scores_22 = []
attention_scores_23 = []

attention_scores_30 = []
attention_scores_31 = []
attention_scores_32 = []
attention_scores_33 = []

attention_scores_40 = []
attention_scores_41 = []
attention_scores_42 = []
attention_scores_43 = []

attention_scores_50 = []
attention_scores_51 = []
attention_scores_52 = []
attention_scores_53 = []

attention_scores_60 = []
attention_scores_61 = []
attention_scores_62 = []
attention_scores_63 = []

attention_scores_70 = []
attention_scores_71 = []
attention_scores_72 = []
attention_scores_73 = []

attention_scores_cls = []


for i, (img, labels) in enumerate(testloader):
    print('Predicting image', i)
    img = img.to(device)
    preds, attns = v(img)
    # Append attention scores of the CLS token to the list
    attention_scores_cls.append(attns[0][:, :, 0, 1:].cpu().detach().numpy())  # batch x depth x head x token

# Convert the list to a NumPy array for further manipulation
attention_scores_cls = np.array(attention_scores_cls)

# Calculate the mean along axis 0 to average attention scores
average_attention_scores_cls = np.mean(attention_scores_cls, axis=0)

# Reshape the array for subplots
reshaped_attention_scores_cls = average_attention_scores_cls.reshape(8, 4, 3, 3)

total_attention=[]
# Plot the attention scores of the CLS token to the rest of the tokens in the same plot
fig, axes = plt.subplots(4, 4, figsize=(20, 20))

for i in range(4):
    for j in range(4):
        axes[i, j].imshow(reshaped_attention_scores_cls[i, j], cmap='viridis')
        axes[i, j].axis('off')

        
# Add common labels to x-axis and y-axis with larger text
fig.text(0.5, 0.07, 'Depth (encoders stacked in series)', ha='center', fontsize=25, fontname='Helvetica')  # Adjust fontsize
fig.text(0.07, 0.5, 'Heads (in parallel)', va='center', rotation='vertical', fontsize=25, fontname='Helvetica')  # Adjust fontsize

plt.savefig('./figures/attention_weights_rgs_token.png')
plt.show()

total_attention = np.sum(reshaped_attention_scores_cls, axis=(0, 1))/2

# Continue with a second figure of the total_attention
plt.figure(figsize=(20, 20))
plt.axis('off')
im = plt.imshow(total_attention, cmap='viridis', vmin=0)
cbar = plt.colorbar(im)
cbar.ax.tick_params(labelsize=20)  # Adjust the font size of colorbar numbers
cbar.ax.yaxis.get_offset_text().set_fontsize(15)  # Adjust the font size of the colorbar offset text
cbar.ax.yaxis.get_offset_text().set_fontname('Helvetica')  # Adjust the font name of the colorbar offset textplt.axis('off')
plt.savefig('./figures/total_attention_scores_rgs.png')
plt.show()



# # Display the average attention scores of CLS to Tokens
# plt.figure(figsize=(8, 8))
# plt.imshow(np.mean(average_attention_scores_cls.reshape(3,3), axis=(0, 1)), cmap='viridis')
# plt.colorbar()
# plt.title('Average Attention Scores of CLS to Tokens')
# plt.axis('off')
# plt.savefig(f'./figures/average_attention_scores.png')
#     attention_scores_00.append(attns[0][0][0][0].cpu().detach().numpy()) # batch x depth x head x token

#     attention_scores_00.append(attns[0][0][0][0].cpu().detach().numpy())
#     attention_scores_01.append(attns[0][1][0][0].cpu().detach().numpy())
#     attention_scores_02.append(attns[0][2][0][0].cpu().detach().numpy())
#     attention_scores_03.append(attns[0][3][0][0].cpu().detach().numpy())

#     attention_scores_10.append(attns[0][0][1][0].cpu().detach().numpy())
#     attention_scores_11.append(attns[0][1][1][0].cpu().detach().numpy())
#     attention_scores_12.append(attns[0][2][1][0].cpu().detach().numpy())
#     attention_scores_13.append(attns[0][3][1][0].cpu().detach().numpy())

#     attention_scores_20.append(attns[0][0][2][0].cpu().detach().numpy())
#     attention_scores_21.append(attns[0][1][2][0].cpu().detach().numpy())
#     attention_scores_22.append(attns[0][2][2][0].cpu().detach().numpy())
#     attention_scores_23.append(attns[0][3][2][0].cpu().detach().numpy())

#     attention_scores_30.append(attns[0][0][3][0].cpu().detach().numpy())
#     attention_scores_31.append(attns[0][1][3][0].cpu().detach().numpy())
#     attention_scores_32.append(attns[0][2][3][0].cpu().detach().numpy())
#     attention_scores_33.append(attns[0][3][3][0].cpu().detach().numpy())

#     attention_scores_40.append(attns[0][0][4][0].cpu().detach().numpy())
#     attention_scores_41.append(attns[0][1][4][0].cpu().detach().numpy())
#     attention_scores_42.append(attns[0][2][4][0].cpu().detach().numpy())
#     attention_scores_43.append(attns[0][3][4][0].cpu().detach().numpy())

#     attention_scores_50.append(attns[0][0][5][0].cpu().detach().numpy())
#     attention_scores_51.append(attns[0][1][5][0].cpu().detach().numpy())
#     attention_scores_52.append(attns[0][2][5][0].cpu().detach().numpy())
#     attention_scores_53.append(attns[0][3][5][0].cpu().detach().numpy())

#     attention_scores_60.append(attns[0][0][6][0].cpu().detach().numpy())
#     attention_scores_61.append(attns[0][1][6][0].cpu().detach().numpy())
#     attention_scores_62.append(attns[0][2][6][0].cpu().detach().numpy())
#     attention_scores_63.append(attns[0][3][6][0].cpu().detach().numpy())

#     attention_scores_70.append(attns[0][0][7][0].cpu().detach().numpy())
#     attention_scores_71.append(attns[0][1][7][0].cpu().detach().numpy())
#     attention_scores_72.append(attns[0][2][7][0].cpu().detach().numpy())
#     attention_scores_73.append(attns[0][3][7][0].cpu().detach().numpy())



# # Convert the list to a NumPy array for further manipulation
# attention_scores = np.array(attention_scores)

# # Calculate the mean along axis 0 to average attention scores
# average_attention_scores = np.mean(attention_scores, axis=0)

# print(np.round(average_attention_scores,2))

# average_attention_scores_excluding_cls = average_attention_scores[:, 1:]

# # Assuming `average_attention_scores` is your final average array
# num_arrays = average_attention_scores_excluding_cls.shape[0]

# # Reshape each array into a 3x3 array of arrays
# reshaped_arrays = average_attention_scores_excluding_cls.reshape(num_arrays, 3, 3)

# # Save the reshaped arrays as images in the "figures" folder
# for i in range(num_arrays):
#     # Plot and save the reshaped array
#     plt.figure()
#     plt.imshow(reshaped_arrays[i], cmap='viridis')
#     plt.colorbar()
#     plt.title(f'Patch {i} (0 is CLS)')
#     plt.axis('off')
#     plt.savefig(f'./figures/attention_scores{i}.png')
#     plt.close()


