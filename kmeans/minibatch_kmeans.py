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

def kmeans(data, clusters=400, save_name='test.png', epochs=30, batch_size=300):

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
    for i in xrange(D_tmp.shape[1]):
        D_tmp[:, i] = data_norm[:, numpy.random.randint(0, data_norm.shape[1])]
    #climin.initialize.randomize_normal(D_tmp, 120, 50)

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

    c = theano.shared(numpy.zeros(clusters, dtype=theano.config.floatX),
                      name='c',
                      borrow=True
                      )

    # setting up the functions to begin looping
    # nearest cluster
    x = T.dmatrix()
    nearest, _ = theano.scan(
        fn=lambda x, D: T.argmax(T.sum(T.sqr(D.T - x), axis=1)),
        outputs_info=None,
        sequences=X.T,
        non_sequences=D
    )

    def update_cluster(x, number, D, c):
        c = T.inc_subtensor(c[number], 1)
        learning_rate = 1/c[number]
        D = T.set_subtensor(D[:, number], D[:, number] * (1 - learning_rate) + x * learning_rate)
        return [D, c]

    ([D_new, c_new], updates) = theano.scan(
        fn=update_cluster,
        sequences=[X.T, nearest],
        outputs_info=[D, c]
    )

    train = theano.function([index], outputs=[D_new, c_new], givens={X: X[:, index * batch_size: (index + 1) * batch_size]})

    # setting up the functions to calculate cost
    S_tmp = theano.shared(numpy.zeros((clusters, data_norm.shape[1]), dtype=theano.config.floatX),
                      name="S",
                      borrow=True
                      )


    print("[*] Training ...")

    start_time = timeit.default_timer()

    for i in xrange(epochs):
        for b in xrange(n_minibatch):
            D_res, c_res = train(b)
            D.set_value(D_res[-1])
            c.set_value(c_res[-1])

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
    kmeans(data, epochs=3, clusters=400, save_name='cifar_minibatch.png')