from __future__ import print_function

import numpy
import theano
import theano.tensor as T
import timeit
import os
import sys
import climin
import climin.initialize
import climin.util
import six.moves.cPickle as pickle

from data import load_data
from MLP import MLP


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=300, optimizer='rmsprop', verbose=False,
             filename='best_model.pkl'):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    Made compatible with climin

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset
    """
    # theano.config.optimizer='None'
    # theano.config.exception_verbosity='high'

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('[*] building the model ...')

    # define a unpack function for the flat parameters
    def unpack(parameters, templates):
        pars = []
        pos = 0
        for t in templates:
            size = [numpy.prod(i) for i in t]
            size = sum(size)
            pars.append(climin.util.shaped_from_flat(parameters[pos:pos+size], t))
            pos += size
        return pars

    # CLIMIN
    wrt = numpy.zeros(28*28*n_hidden + n_hidden + n_hidden * 10 + 10, dtype=theano.config.floatX)
    templates = [[(28*28, n_hidden), n_hidden], [(n_hidden, 10), 10]]

    # initialize weights, not biases
    for p in unpack(wrt, templates):
        climin.initialize.randomize_normal(p[0], 0, 0.01)

    # datastream
    args = ((i, {}) for i in climin.util.iter_minibatches(
        [train_set_x.eval(), train_set_y.eval()], batch_size, [0, 0]))

    # THEANO
    # allocate symbolic variables for the data
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    pars = []
    for p in unpack(wrt, templates):
        pars.append((theano.shared(value=p[0], borrow=True), theano.shared(value=p[1], borrow=True)))

    # construct the MLP class
    classifier = MLP(
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10,
        W=[pars[0][0], pars[1][0]],
        b=[pars[0][1], pars[1][1]]
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_error = theano.function(
        inputs=[],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x,
            y: test_set_y
        }
    )

    val_error = theano.function(
        inputs=[],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x,
            y: valid_set_y
        }
    )

    train_error = theano.function(
        inputs=[],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x,
            y: valid_set_y
        }
    )

    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    grad = theano.function(
        inputs=[x, y],
        outputs=gparams
    )

    def d_loss_wrt_pars(parameters, inpt, targets):
        set_pars(parameters)
        n = numpy.array([g for g in grad(inpt, targets)])
        return numpy.concatenate((n[0].flatten(), n[1], n[2].flatten(), n[3]))

    def set_pars(parameters):
        for tp, cp in zip(classifier.params, [i for sub in unpack(parameters, templates) for i in sub]):
            tp.set_value(cp, borrow=True)

    # build optimizer
    if optimizer == 'gradient_descent':
        print("[*] using GRADIENT DESCENT ...")
        opt = climin.GradientDescent(wrt, d_loss_wrt_pars, step_rate=0.1, momentum=0.0, momentum_type='nesterov',
                                     args=args)
    elif optimizer == 'rmsprop':
        print("[*] using RMSPROP ...")
        opt = climin.RmsProp(wrt, d_loss_wrt_pars, step_rate=0.001, decay=0.9, momentum=0, step_adapt=False,
                             step_rate_min=0, step_rate_max=numpy.inf, args=args)

    ###############
    # TRAIN MODEL #
    ###############
    print('[*] training ...')


    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_loss = 0.
    start_time = timeit.default_timer()

    va_losses = []
    tr_losses = []
    te_losses = []
    x_axis = []

    best_params = numpy.empty_like(wrt)
    best_test = numpy.inf

    for info in opt:
        iteration = info['n_iter']
        epoch = iteration // n_train_batches
        minibatch_index = iteration % n_train_batches

        if iteration % validation_frequency == 0:
            # compute zero-one loss on validation set
            validation_loss = val_error()

            if verbose:
                va_losses.append(validation_loss)
                te_losses.append(test_error())
                tr_losses.append(train_error())
                x_axis.append(iteration / n_train_batches)

            print(
                "epoch %i, minibatch %i/%i, validation error %f %%" %
                (
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    validation_loss * 100.
                )
            )

            # if we got the best validation score until now
            if validation_loss < best_validation_loss:
                # improve patience if loss improvement is good enough
                if validation_loss < best_validation_loss * improvement_threshold:
                    patience = max(patience, iteration * patience_increase)

                best_validation_loss = validation_loss

                best_params[...] = wrt
                test_loss = test_error()
                best_epoch = epoch

                print(
                    "epoch %i, minibatch %i/%i, test error of best model %f %%" %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        test_loss * 100.
                    )
                )

        if patience <= iteration or epoch >= n_epochs:
            # set parameters in the LogReg class to make sure they are up to date
            set_pars(wrt)
            break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_epoch, test_loss * 100.))
    print(('The code for file ' +
            os.path.split(__file__)[1] +
           ' ran for %.2fm, with %f epochs/min' % ((end_time - start_time) / 60., epoch/(end_time - start_time) * 60.)),
          file=sys.stderr)


if __name__ == '__main__':
    test_mlp()