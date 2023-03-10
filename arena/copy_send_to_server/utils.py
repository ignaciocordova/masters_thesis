import pandas as pd
import numpy as np

import torch
import torch.nn as nn


def get_data_and_target(df, target_df, coordenates_of_interest, channels, normalize_target=False, INSTALLED_POWER=17500):
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
    normalize_target : bool, optional
        If True, the target is normalized dividing by the maximum power. The default is False.
    
    Returns
    -------
    data_of_interest : torch tensor
        Shape: (n_samples, n_channels, 9, 9)
        Tensor with all the images of the coordinates of interest.
    target_of_interest : torch tensor
        Shape: (n_samples)
        Tensor with the target of interest.

    """
    # filter out unwanted columns to keep only the coordinates of interest
    df = df.filter(regex='|'.join(coordenates_of_interest))

    # create images
    data_of_interest = create_images(df, channels, image_size=9)
    
    # normalize target if needed
    target = target_df[1].values.astype(np.float32)
    if normalize_target:
        target /= INSTALLED_POWER

    return data_of_interest, torch.from_numpy(target)


def create_images(df, channels, image_size):
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
        Shape: (n_samples, n_channels, image_size, image_size)
        Tensor with all the images of the coordinates of interest.

    """
    images = np.zeros((df.shape[0], len(channels), image_size, image_size), dtype=np.float32)

    for i, channel in enumerate(channels):
        channel_data = df.filter(regex=channel).values.reshape(df.shape[0], image_size, image_size)
        images[:, i, :, :] = channel_data

    return torch.from_numpy(images)

