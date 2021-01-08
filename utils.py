
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



def plot_time_series_class(data, class_name, ax, n_steps=10):
      time_series_df = pd.DataFrame(data)

      smooth_path = time_series_df.rolling(n_steps).mean()
      path_deviation = 2 * time_series_df.rolling(n_steps).std()

      under_line = (smooth_path - path_deviation)[0]
      over_line = (smooth_path + path_deviation)[0]

      ax.plot(smooth_path, linewidth=2)
      ax.fill_between(
        path_deviation.index,
        under_line,
        over_line,
        alpha=.125
      )
      ax.set_title(class_name)

```
classes = df.target.unique()

fig, axs = plt.subplots(
  nrows=len(classes) // 3 + 1, #divide by 3 and add +1
  ncols=3,
  sharey=True,#share y axis
  figsize=(14, 8)
)

for i, cls in enumerate(classes):
  ax = axs.flat[i]
  data = df[df.target == cls] \ # filter the target class in each loop
    .drop(labels='target', axis=1) \#drop target coln
    .mean(axis=0) \# take mean value for each coln
    .to_numpy()
  plot_time_series_class(data, class_names[i], ax)

fig.delaxes(axs.flat[-1])# delete last axis as we have only five targets
fig.tight_layout();
```


def train_model(model, train_dataset, val_dataset, n_epochs):
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.L1Loss(reduction='sum').to(device)
  history = dict(train=[], val=[])

  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0

  for epoch in range(1, n_epochs + 1):
    model = model.train()

    train_losses = []
    for seq_true in train_dataset:
      optimizer.zero_grad()

      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)

      loss = criterion(seq_pred, seq_true)

      loss.backward()
      optimizer.step()

      train_losses.append(loss.item())

    val_losses = []
    model = model.eval()
    with torch.no_grad():
      for seq_true in val_dataset:

        seq_true = seq_true.to(device)
        seq_pred = model(seq_true)

        loss = criterion(seq_pred, seq_true)
        val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    history['train'].append(train_loss)
    history['val'].append(val_loss)

    if val_loss < best_loss:
      best_loss = val_loss
      torch.save(model.state_dict(), "best_model.pt")

    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

    print("Best val_loss: {:4f}".format(best_loss))

  model.load_state_dict(best_model_wts)
  return model.eval(), history

def plot_hist(history):
    ax = plt.figure().gca()

    ax.plot(history['train'])
    ax.plot(history['val'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'])
    plt.title('Loss over training epochs')
    plt.show();
