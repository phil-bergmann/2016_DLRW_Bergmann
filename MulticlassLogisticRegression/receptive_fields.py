from __future__ import print_function

import matplotlib.pyplot as plt
import six.moves.cPickle as pickle
import numpy
import matplotlib.image as mpimg


def receptive_flieds(model='rmsprop.pkl'):

    # load the saved model
    classifier = pickle.load(open(model))
    W = classifier.W.get_value(borrow=True)

    # get the pixels per digit shaped 28x28
    W = W.transpose((1, 0)).reshape(10, 28, 28)

    figure, (axes) = plt.subplots(2, 5)
    axes = axes.flatten()

    for i in xrange(W.shape[0]):
        axes[i].imshow(W[i], cmap='gray')
        axes[i].set_title("Weights for %d" % i, size='medium')
        axes[i].axis('off')

    figure.set_facecolor('white')
    figure.suptitle(model.split('.')[0], size='xx-large')
    figure.savefig('repflds.png')
    plt.show()


if __name__ == '__main__':
    receptive_flieds(model='rmsprop.pkl')
    receptive_flieds(model='adam.pkl')