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


parser = argparse.ArgumentParser(description='plot image generated')
parser.add_argument('file', action='store', nargs='*', 
                    help='model files to plot')
parser.add_argument('-ld', '--latent', action='store', type=int,  default=32, 
                    help='latent dimension in autoencoder (must be similar to model)')
parser.add_argument('-n', '--number', action='store', type=int, default=5, 
                    help='number of epoch to do')

args = parser.parse_args()

file_name = args.file
latent_dimension = args.latent
n = args.number

if len(file_name)==0:
  print 'Error : no files where parse'
  sys.exit()
for f in file_name:
  if os.path.isfile(f)==False:
    print 'Error: file '+f+' does not exist'
    sys.exit()


from keras.models import load_model
from mpl_toolkits.axes_grid1 import ImageGrid

model = load_model(file_name[0])
Z = np.mgrid[2:-2.2:-0.2, -2:2.2:0.2].reshape(latent_dimension, -1).T[:, ::-1].astype(np.float32)
batch_size = 128

reconstructions = []
print('generating samples')
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
plt.title('samples generated from latent space of autoencoder')
grid = ImageGrid(
    fig, 111, nrows_ncols=(21, 21),
    share_all=True)

print('plotting latent space')
for i, x in enumerate(X):
    img = (x * 255).astype(np.uint8)
    grid[i].imshow(img, cmap='Greys_r')
    grid[i].get_xaxis().set_visible(False)
    grid[i].get_yaxis().set_visible(False)
    grid[i].set_frame_on(False)

plt.savefig('latent_train_val.png', bbox_inches='tight')


for f in file_name:
  print 'loading', f

  model = load_model(f)

  noise = np.random.uniform(0, 1, size=[n,latent_dimension])

  print 'decoding images'

  imgs = model.predict(noise)

  plt.figure()
  for i in range(n):    
    ax = plt.subplot(2, n, i+1)
    plt.imshow(noise[i].reshape(latent_dimension, 1))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
 
  plt.savefig('noise_'+f[:-2]+'png')

  print 'noise_'+f[:-2]+'png'

  plt.figure()
  for i in range(n):    
    ax = plt.subplot(2, n, i+1)
    plt.imshow(imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
 
  plt.savefig(f[:-2]+'png')

  print f[:-2]+'png'

