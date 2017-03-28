import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import argparse
import sys
import os



def get_batch_idx(N, batch_size):
    num_batches = (N + batch_size - 1) / batch_size

    for i in range(num_batches):
        start, end = i * batch_size, (i + 1) * batch_size
        idx = slice(start, end)

        yield idx

from keras.models import load_model
from mpl_toolkits.axes_grid1 import ImageGrid

def latent_space(model, fname, latent_dimension):
  Z = np.mgrid[2:-2.2:-0.2, -2:2.2:0.2].reshape(latent_dimension, -1).T[:, ::-1].astype(np.float32)
  batch_size = 128

  reconstructions = []
  for idx in get_batch_idx(Z.shape[0], batch_size):
    Z_batch = Z[idx]
    X_batch = model.predict(Z_batch)
    reconstructions.append(X_batch)

  X = np.vstack(reconstructions)
  X = X.reshape(X.shape[0], 28, 28)

  fig = plt.figure(1, (12., 12.))
  ax1 = plt.axes(frameon=False)
  ax1.get_xaxis().set_visible(False)
  ax1.get_yaxis().set_visible(False)
  #plt.title('samples generated from latent space of autoencoder')
  grid = ImageGrid(
      fig, 111, nrows_ncols=(21, 21),
      share_all=True)

  #print('plotting latent space')
  for i, x in enumerate(X):
    img = (x * 255).astype(np.uint8)
    grid[i].imshow(img, cmap='Greys_r')
    grid[i].get_xaxis().set_visible(False)
    grid[i].get_yaxis().set_visible(False)
    grid[i].set_frame_on(False)

  plt.savefig(fname, bbox_inches='tight')
