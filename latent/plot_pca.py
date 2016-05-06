from __future__ import print_function


import matplotlib.pyplot as plt
from itertools import product

from data import load_data
from data import load_cifar
from pca import pca


def plot_pca(dataset, save_name='test.png', classes=10, labels=None):

    dataset_x, dataset_y = dataset

    print("[*] plotting ...")

    figure, (axes) = plt.subplots(classes, classes)
    figure.set_size_inches(50, 50)
    plt.prism()

    # set labels
    if labels:
        for ax, lb in zip(axes[0], labels):
            ax.set_title(lb, size=16)
        for ax, lb in zip(axes[:, 0], labels):
            ax.set_ylabel(lb, size=16)
    else:
        for i in xrange(classes):
            axes[0, i].set_title(i, size='large')
        for i in xrange(classes):
            axes[i, 0].set_ylabel(i, size='large')

    # plot the data
    for i, j in product(xrange(10), repeat=2):
        if i > j:
            continue
        X_ = pca(dataset_x[(dataset_y == i) + (dataset_y == j)], 2)
        y_ = dataset_y[(dataset_y == i) + (dataset_y == j)]

        axes[i, j].scatter(X_[:, 0], X_[:, 1], c=y_)
        axes[i, j].set_xticks(())
        axes[i, j].set_yticks(())

        axes[j, i].scatter(X_[:, 0], X_[:, 1], c=y_)
        axes[j, i].set_xticks(())
        axes[j, i].set_yticks(())


    #figure.suptitle("MNIST", size='xx-large')
    plt.tight_layout()
    plt.savefig(save_name)


if __name__ == '__main__':
    # MNIST
    mnist = load_data('mnist.pkl.gz', shared=False)
    plot_pca(mnist[0], 'scatterplotMNIST.png')

    # CIFAR-10
    cifar = load_cifar(concat=True, label_names=True)
    plot_pca(cifar[0], 'scatterplotCIFAR.png', labels=cifar[2])