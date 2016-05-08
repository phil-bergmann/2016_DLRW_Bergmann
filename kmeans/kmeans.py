import numpy
import theano
import theano.tensor as T
import climin.initialize

from data import load_cifar

def test(data, clusters=200):

    # 1. Normalize inputs
    data_norm = (data - numpy.mean(data, axis=1)[:, numpy.newaxis])
    data_norm = data_norm / numpy.sqrt(numpy.var(data, axis=1) + 10)[:, numpy.newaxis]

    # transpose data for further processing in according to the paper
    data_norm = data_norm.T


    # 2. Whiten inputs
    epsilon = 0.05
    d, V = numpy.linalg.eig(numpy.cov(data_norm))
    # Eigenvalue Matrix
    #D = numpy.eye(d.shape[0])
    #D = d*D
    #transformation = numpy.dot(numpy.dot(V, numpy.diag(1.0/numpy.sqrt(numpy.diag(d) + epsilon))), V.T)
    transformation = numpy.dot(numpy.dot(V, 1.0 / numpy.sqrt(numpy.diag(d) + epsilon)), V.T)

    # apply transformation
    data_norm = numpy.dot(transformation, data_norm)


    # 3. Loop until convergence
    # initialize cluster centers
    D_tmp = numpy.zeros((data_norm.shape[0], clusters), dtype=theano.config.floatX)
    climin.initialize.randomize_normal(D_tmp, 0, 1)
    # normalize
    D_tmp /= numpy.sum(D_tmp, axis=0)[numpy.newaxis, :]

    S = theano.shared(numpy.zeros((data_norm.shape[0], clusters), dtype=theano.config.floatX),
                      name="S",
                      borrow=True
                      )

    D = theano.shared(D_tmp,
                      name="D",
                      borrow=True
                      )




if __name__ == '__main__':
    data = load_cifar(small=True)[0][0]
    test(data)