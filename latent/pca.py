from __future__ import print_function

import numpy
import numpy.linalg
import theano
import theano.tensor as T
import theano.tensor.nlinalg

def pca(dataset, count):

    # calculate scatter matrix
    tmp = numpy.cov(dataset.T)
    scatter = theano.shared(tmp, borrow=True)

    # calculate eigenvalues / -vectors
    eigen_values, eigen_vectors = theano.tensor.nlinalg.eig(scatter)

    # choose the <count> eigenvectors with the biggest eigenvalues and compute dataset representation in subspace
    result = T.dot(dataset, eigen_vectors[:, :count])
    compute_result = theano.function(
        inputs=[],
        outputs=result
    )

    return compute_result()