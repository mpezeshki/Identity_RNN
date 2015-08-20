import numpy as np
from utils import load_data
from fuel.datasets import IterableDataset
from fuel.streams import DataStream
import theano

floatX = theano.config.floatX


def MNIST(batch_size=200):
    train_set, valid_set, test_set = load_data('mnist.pkl.gz')
    train_x = train_set[0].reshape((50000 / batch_size, batch_size, 784))
    train_x = np.swapaxes(train_x, 2, 1)
    train_x = train_x[:, :, :, np.newaxis]
    # Now the dimension is num_batches x 784 x batch_size x 1

    train_y = (np.zeros(train_set[0].shape) - 1)
    # label for each time-step is -1 and for the last one is the real label
    train_y[:, -1] = train_set[1]
    train_y = train_y.reshape((50000 / batch_size, batch_size, 784))
    train_y = np.swapaxes(train_y, 2, 1)
    train_y = train_y[:, :, :, np.newaxis]
    # Now the dimension is num_batches x 784 x batch_size x 1

    valid_x = valid_set[0].reshape((10000 / batch_size, batch_size, 784))
    valid_x = np.swapaxes(valid_x, 2, 1)
    valid_x = valid_x[:, :, :, np.newaxis]
    # Now the dimension is num_batches x 784 x batch_size x 1

    valid_y = (np.zeros(valid_set[0].shape) - 1)
    # label for each time-step is -1 and for the last one is the real label
    valid_y[:, -1] = valid_set[1]
    valid_y = valid_y.reshape((10000 / batch_size, batch_size, 784))
    valid_y = np.swapaxes(valid_y, 2, 1)
    valid_y = valid_y[:, :, :, np.newaxis]
    # Now the dimension is num_batches x 784 x batch_size x 1

    train = IterableDataset({'x': train_x.astype(floatX),
                             'y': train_y.astype('int32')})
    train_stream = DataStream(train)

    valid = IterableDataset({'x': valid_x.astype(floatX),
                             'y': valid_y.astype('int32')})
    valid_stream = DataStream(valid)

    return train_stream, valid_stream
