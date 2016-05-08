import numpy

from data import load_data
from run_bhtsne import run_bhtsne

def bhtsne_mnist():
    datasets = load_data('mnist.pkl.gz', shared=False)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    data = numpy.vstack((train_set_x, valid_set_x, test_set_x))

    run_bhtsne(data, save_name='bhtsne_mnist.png')


if __name__ == '__main__':
    bhtsne_mnist()