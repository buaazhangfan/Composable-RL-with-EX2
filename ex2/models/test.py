import tensorflow as tf
import numpy as np 
from siamese_tf import Siamese

import matplotlib.pyplot as plt

def sample_batch(data, data_size, batch_size):
    idxs = np.random.randint(data_size, size=batch_size)

    return data[idxs]

train_itrs=40000
batch_size=512
replay_size = 100000
replay = np.concatenate([np.random.randn(replay_size // 2) - 4, np.random.randn(replay_size // 2) + 4])
replay = np.expand_dims(replay, 1).astype(np.float32)
siamese = Siamese(1, 4, (32, 32), learning_rate=1e-3)
positives_np = np.expand_dims(np.linspace(-8, 8, 200).astype(np.float32), 1)
positives = positives_np
labels = np.expand_dims(np.concatenate([np.ones(batch_size), np.zeros(batch_size)]), 1).astype(np.float32)
hist, bin_edges = np.histogram(replay, density=True, bins=100)
bin_edges += (bin_edges[1] - bin_edges[0]) / 2
log_step = 0.01 * train_itrs
plt.ion()
siamese.build_graph()

siamese.init_tf_sess()

for train_itr in range(train_itrs):

    pos = sample_batch(positives, positives.shape[0], batch_size)
    # print(pos.shape)
    # exit()
    neg = sample_batch(replay, replay.shape[0], batch_size)
    x1 = np.concatenate([pos, pos])
    x2 = np.concatenate([pos, neg])

    loss = siamese.train(x1, x2, labels)
    if train_itr % log_step == 0:
        print(loss)
        pred = siamese.predict(positives, positives)
        plt.clf()
        plt.plot(bin_edges[:-1], hist)
        plt.plot(positives_np, pred)
        plt.show()
        plt.pause(0.05)