import pandas as pd
import numpy as np

import torch
import torch.nn as nn


def get_data_and_target(df, target_df, coordinates_of_interest, channels, normalize_target=False, INSTALLED_POWER=17500):
    """
    Returns data of interest from df and target_df

    Parameters
    ----------
    df : pandas dataframe
        Dataframe with all the data.
    target_df : pandas dataframe
        Dataframe with all the targets.
    coordinates_of_interest : list
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
    df = df.filter(regex='|'.join(coordinates_of_interest))

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



def create_images_with_previous_label_channel(df, target_df, channels, image_size, normalize_target=False, INSTALLED_POWER=17500):
    """
    Creates images from the data in df.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe with all the data.
    target_df : pandas dataframe
        Dataframe with all the targets.
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
    images = np.zeros((df.shape[0], len(channels)+1, # to account for previous label channel
                        image_size, image_size), dtype=np.float32)

    for i, channel in enumerate(channels):
        channel_data = df.filter(regex=channel).values.reshape(df.shape[0], image_size, image_size)
        images[:, i, :, :] = channel_data

    # add previous label channel
    target = target_df[1].values.astype(np.float32)
    if normalize_target:
        target /= INSTALLED_POWER

    previous_label = np.zeros_like(target)
    previous_label[1:] = target[:-1]
    previous_label[0] =  np.mean(target) # uninformed first label !  

    for idx,label in enumerate(previous_label):
        images[idx, -1, :, :] = np.full((image_size, image_size), label)
        # images[idx, -1, :, :] = np.full((image_size, image_size), 0.0) # non-informative

    return torch.from_numpy(images)

def get_data_and_target_with_previous_label_channel(df, target_df, coordinates_of_interest, channels, image_size, normalize_target=False, INSTALLED_POWER=17500):
    """
    Returns data of interest from df and target_df

    Parameters
    ----------
    df : pandas dataframe
        Dataframe with all the data.
    target_df : pandas dataframe
        Dataframe with all the targets.
    coordinates_of_interest : list
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
    df = df.filter(regex='|'.join(coordinates_of_interest))

    # create images
    data_of_interest = create_images_with_previous_label_channel(df, target_df, channels, image_size, normalize_target, INSTALLED_POWER)
    
    # normalize target if needed
    target = target_df[1].values.astype(np.float32)
    if normalize_target:
        target /= INSTALLED_POWER

    return data_of_interest, torch.from_numpy(target)

def create_images_with_all_coordinates(df, channels):
    """
    Creates images from the data in df.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe with all the data.
    channels : list
        List with the channel names to build the image.

    Returns
    -------
    images : torch tensor
        Shape: (n_samples, n_channels, 15, 27)
        Tensor with all the images of the coordinates of interest.

    """
    images = np.zeros((df.shape[0], len(channels), 15, 27), dtype=np.float32)

    for i, channel in enumerate(channels):
        channel_data = df.filter(regex=channel).values.reshape(df.shape[0], 15, 27)
        images[:, i, :, :] = channel_data

    return torch.from_numpy(images)

def get_data_and_target_with_all_coordinates(df, target_df, coordinates_of_interest, channels, normalize_target=False, INSTALLED_POWER=17500):
    """
    Returns data of interest from df and target_df

    Parameters
    ----------
    df : pandas dataframe
        Dataframe with all the data.
    target_df : pandas dataframe
        Dataframe with all the targets.
    coordinates_of_interest : list
        List with the coordinates of interest.
    channels : list
        List with the channel names to build the image.
    normalize_target : bool, optional
        If True, the target is normalized dividing by the maximum power. The default is False.
    
    Returns
    -------
    data_of_interest : torch tensor
        Shape: (n_samples, n_channels, 15, 27)
        Tensor with all the images of the coordinates of interest.
    target_of_interest : torch tensor
        Shape: (n_samples)
        Tensor with the target of interest.

    """
    # filter out unwanted columns to keep only the coordinates of interest
    df = df.filter(regex='|'.join(coordinates_of_interest))

    # create images
    data_of_interest = create_images_with_all_coordinates(df, channels)
    
    # normalize target if needed
    target = target_df[1].values.astype(np.float32)
    if normalize_target:
        target /= INSTALLED_POWER

    return data_of_interest, torch.from_numpy(target)


def create_video_dataset(dataloader, NUM_FRAMES, OVERLAP_SIZE):
    """
    Creates a video dataset from an image dataloader with batch_size=1.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Dataloader for the dataset.
    NUM_FRAMES : int
        Number of frames to build each video.
    OVERLAP_SIZE : int
        Number of frames that overlap between consecutive videos.

    Returns
    -------
    List
        A list containing the video dataset. Each element of the list is a tuple, containing a video tensor of shape 
        (NUM_FRAMES, C, H, W) and a label tensor of shape (1), where C, H, and W are the number of channels, height, 
        and width of each image in the dataset, respectively.
    """

    video_dataset = []
    video = []
    for i, (image, label) in enumerate(dataloader):
        video.extend(image)
        if len(video) == NUM_FRAMES:
            for j in range(NUM_FRAMES - OVERLAP_SIZE):
                video_dataset.append((torch.stack(video[j:j+NUM_FRAMES]), label))
            video = video[NUM_FRAMES - OVERLAP_SIZE:]

    # handle the remaining frames that don't form a complete video
    if len(video) > 0:
        for j in range(len(video) - NUM_FRAMES + 1):
            video_dataset.append((torch.stack(video[j:j+NUM_FRAMES]), label))

    return video_dataset