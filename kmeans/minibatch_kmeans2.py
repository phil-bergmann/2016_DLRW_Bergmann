from __future__ import print_function

import numpy
import theano
import theano.tensor as T
import climin.initialize
import timeit
import os
import sys
import math
from PIL import Image

from data import load_cifar
from tile_raster_images import tile_raster_images

def kmeans(data, batch_size, clusters=400, save_name='test.png', epochs=30, damping=1):

    n_minibatch = data.shape[0] / batch_size

    print("[*] Initializing ...")

    # 1. Normalize inputs
    data_norm = (data - numpy.mean(data, axis=1)[:, numpy.newaxis])
    data_norm = data_norm / numpy.sqrt(numpy.var(data, axis=1) + 10)[:, numpy.newaxis]

    # transpose data for further processing in according to the paper
    data_norm = data_norm.T


    # 2. Whiten inputs
    d, V = numpy.linalg.eig(numpy.cov(data_norm))
    epsilon = 0.1 * numpy.eye(d.shape[0])
    transformation = numpy.dot(numpy.dot(V, numpy.linalg.inv(numpy.sqrt(numpy.diag(d) + epsilon))), V.T)

    # apply transformation
    data_norm = numpy.dot(transformation, data_norm)


    # 3. Loop until convergence
    index = T.lscalar()
    # initialize cluster centers
    D_tmp = numpy.zeros((data_norm.shape[0], clusters), dtype=theano.config.floatX)
    climin.initialize.randomize_normal(D_tmp, 0, 1)

    D = theano.shared(D_tmp,
                      name="D",
                      borrow=True
                      )

    # copy data to shared var for performance
    X = theano.shared(data_norm.astype(theano.config.floatX))

    S = theano.shared(numpy.zeros((clusters, batch_size), dtype=theano.config.floatX),
                      name="S",
                      borrow=True
                      )

    # setting up the functions to begin looping
    # update S
    big_pos = T.argmax(abs(T.dot(D.T, X)), axis=0)
    big_vals = T.dot(D.T, X)[big_pos, T.arange(big_pos.shape[0])]
    zeros_sub = T.zeros_like(S)[big_pos, T.arange(big_pos.shape[0])]
    S_new = T.set_subtensor(zeros_sub, big_vals)

    update_S = theano.function([index],
                               updates=[(S, S_new)],
                               givens={X: X[:, index * batch_size: (index + 1) * batch_size]}
                               )
    update_D = theano.function([index],
                               updates=[(D, T.dot(X, S.T) + damping*D)],
                               givens={X: X[:, index * batch_size: (index + 1) * batch_size]}
                               )
    normalize_D = theano.function([], updates=[(D, D / T.sqrt(T.sum(T.sqr(D), axis=0)))])
    cost = theano.function([index],
                           outputs=T.sum(T.sqrt(T.sum(T.sqr(T.dot(D, S) - X), axis=0))),
                           givens={X: X[:, index * batch_size: (index + 1) * batch_size]}
                           )

    normalize_D()

    print("[*] Training ...")

    start_time = timeit.default_timer()

    for i in xrange(epochs):
        for b in xrange(n_minibatch):
            update_S(b)
            update_D(b)
            normalize_D()
            print('Training epoch %d, minibatch %d, cost %.2f' % (i + 1, b + 1, cost(b)))

    end_time = timeit.default_timer()

    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fs' % (end_time - start_time)), file=sys.stderr)

    image = Image.fromarray(
        tile_raster_images(X=D.get_value(borrow=True).T,
                           img_shape=(12, 12), tile_shape=(int(math.sqrt(clusters)), int(math.sqrt(clusters))),
                           tile_spacing=(1, 1)))
    image.save(save_name)


if __name__ == '__main__':
    data = load_cifar(small=True)[0][0]
    kmeans(data, clusters=400, save_name='cifar_minibatch2.png', batch_size=500, damping=100)