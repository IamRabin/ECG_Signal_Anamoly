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


def main(config):

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
        test_normal_dataset, seq_len, _ = create_dataset(test_df)
        test_anomaly_dataset, _, _ = create_dataset(anomaly_df)

       with torch.set_grad_enabled(False):
                model =  RecurrentAutoencoder(seq_len, n_features, 128)
                state_dict = torch.load(config.weights, map_location=device)
                unet.load_state_dict(state_dict)
                model.eval()
                model.to(device)

               _, losses = predict(model,test_normal_dataset)

               #anamoly val_dataset
               _,pred_losses= predict(model,test_anamoly_dataset)

              THRESHOLD = 26
               correct = sum(l > THRESHOLD for l in pred_losses)
              print(f'Correct anomaly predictions: {correct}/{len(anomaly_dataset)}')


def predict(model, dataset):
  predictions, losses = [], []
  criterion = nn.L1Loss(reduction='sum').to(device)
  with torch.no_grad():
    model = model.eval()
    for seq_true in dataset:
      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)

      loss = criterion(seq_pred, seq_true)

      predictions.append(seq_pred.cpu().numpy().flatten())
      losses.append(loss.item())
  return predictions, losses

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Inference for ECG anamoly detection"
    )
    parser.add_argument(
        "--weights", type=str, default="./weights/best_model.pt", required=True, help="path to weights file"
    )
    config = parser.parse_args()
    main(config)
