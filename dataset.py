

import copy
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from arff2pandas import a2p




def create_dataset(df):
      sequences = df.astype(np.float32).to_numpy().tolist()
      dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
      n_seq, seq_len, n_features = torch.stack(dataset).shape
      return dataset, seq_len, n_features
