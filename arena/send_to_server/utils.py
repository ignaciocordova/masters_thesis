import pandas as pd
import numpy as np

import torch
import torch.nn as nn


def get_data_and_target_of_interest(df, target_df, coordenates_of_interest, channels):
    """
    Returns data of interest from df and target_df

    Parameters
    ----------
    df : pandas dataframe
        Dataframe with all the data.
    target_df : pandas dataframe
        Dataframe with all the targets.
    coordenates_of_interest : list
        List with the coordinates of interest.
    channels : list
        List with the channel names to build the image.
    
    Returns
    -------
    data_of_interest : torch tensor
        Tensor with all the images of the coordinates of interest.
    target_of_interest : torch tensor
        Tensor with the target of interest.

    """
    # keep columns that contain any of the coordinates of interest
    df = df.loc[:, df.columns.str.contains('|'.join(coordenates_of_interest))]

    # get data of interest
    data_of_interest = []
    for i in range(df.shape[0]):
        image = []
        for channel in channels: 
            image.append(df.loc[i, df.columns.str.contains(channel)].values.reshape(9,9))
        data_of_interest.append(image)
    
    # transform data to tensor
    data_of_interest = torch.tensor(np.array(data_of_interest, dtype=np.float32))
    
    # get target of interest
    target_of_interest = torch.tensor(np.array(target_df['Target (KWh)'].values, dtype=np.float32))
        
    return target_of_interest, data_of_interest




