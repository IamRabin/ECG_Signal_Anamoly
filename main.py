import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import copy
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from arff2pandas import a2p



from dataset import create_dataset
from model import RecurrentAutoencoder
from utils import train_model

def main():



    with open('ECG_TRAIN.arff') as f:
          train = a2p.load(f)
    with open('ECG_TEST.arff') as f:
          test = a2p.load(f)

    #Merge dataset
    df = train.append(test)
    #shuffling data frame by sampling with frac=1
    df = df.sample(frac=1.0)

    CLASS_NORMAL = 1
    class_names = ['Normal','R on T','PVC','SP','UB']



    normal_df = df[df.target == str(CLASS_NORMAL)].drop(labels='target', axis=1)
    #We'll merge all other classes and mark them as anomalies:
    anomaly_df = df[df.target != str(CLASS_NORMAL)].drop(labels='target', axis=1)

    #We'll split the normal examples into train, validation and test sets:
    train_df, val_df = train_test_split(
      normal_df,
      test_size=0.15,
      random_state=101
    )

    val_df, test_df = train_test_split(
      val_df,
      test_size=0.33,
      random_state=101
    )

    train_dataset, seq_len, n_features = create_dataset(train_df)
    val_dataset, _, _ = create_dataset(val_df)
    test_normal_dataset, _, _ = create_dataset(test_df)
    test_anomaly_dataset, _, _ = create_dataset(anomaly_df)

    if torch.cuda.device_count() >= 1:
        print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))
        model = model.cuda()
    else:
        raise ValueError('CPU training is not supported')

    model = RecurrentAutoencoder(seq_len, n_features, 128)
    model = model.to(device)


    model, history = train_model(
      model,
      train_dataset,
      val_dataset,
      n_epochs=150
    )





if __name__ == '__main__':
    main()
