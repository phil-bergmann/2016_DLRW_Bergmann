from __future__ import print_function

import six.moves.cPickle as pickle
import theano
import theano.tensor as T
import numpy

from logistic_optimizer import opt
from data import load_data

def generate():
    opt(optimizer='adam', save=True, filename='adam.pkl', improvement_threshold=1, patience=80000,
        n_epochs=numpy.inf)
    # rmsprop, 6.65 - 7.28
    # adam 6.69 7.22

def validate(model='adam.pkl', dataset='mnist.pkl.gz'):

    # load the saved model
    classifier = pickle.load(open(model))

    datasets = load_data(dataset)
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    x = T.matrix('x')  # data
    y = T.ivector('y')  # class labels

    test_model = theano.function(
        inputs=[],
        outputs=classifier.errors(y),
        givens={
            classifier.input: test_set_x,
            y: test_set_y
        }
    )

    validate_model = theano.function(
        inputs=[],
        outputs=classifier.errors(y),
        givens={
            classifier.input: valid_set_x,
            y: valid_set_y
        }
    )

    print("[*] loaded '%s'" % model)
    print("Validation Error: %f" % validate_model())
    print("Test Error: %f" % test_model())

if __name__ == '__main__':
    validate()