import climin
import climin.initialize
import climin.util
import numpy
import theano
import theano.tensor as T
import six.moves.cPickle as pickle
import timeit
import os
import sys

from data import load_data
from LogReg import LogReg


def opt(learning_rate=0.13, n_epochs=1000, batch_size=600, dataset='mnist.pkl.gz', optimizer='gd'):
    """ Further optimized training function for the Logistic Regression

    Demonstration on MNIST dataset.

    :type learning_rate: float
    :param learning_rate: learning rate used for stochastic gradient

    :type n_epochs: int
    :param n_epochs: maximal number of epochs run

    :type dataset: string
    :param dataset: the path to the MNIST dataset file

    :type batch_size: int
    :param batch_size: size of the batches used
    """

    # load the data, use numpy arrays not shared theano vars
    datasets = load_data(dataset, False)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute the number of minibatches per epoch
    n_train_batches = 1. * train_set_x.shape[0] / batch_size
    n_valid_batches = 1. * valid_set_x.shape[0] / batch_size
    n_test_batches = 1. * test_set_x.shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print("[*] Building model ...")

    x = T.matrix('x')  # data
    y = T.ivector('y')  # class labels

    classifier = LogReg(input=x, n_in=28 * 28, n_out=10)

    # cost to minimize is negative log-likelihood
    cost = classifier.negative_llh(y)

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # compile functions that measure the mistakes made by the model on each minibatch
    test_model = theano.function(
        inputs=[],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x,
            y: test_set_y,
        }
    )

    validate_model = theano.function(
        inputs=[],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x,
            y: valid_set_y,
        }
    )

    def set_pars(parameters):
        W, b = climin.util.shaped_from_flat(parameters, tmpl)
        classifier.W.set_value(W, borrow=True)
        classifier.b.set_value(b, borrow=True)

    # negative log-likelihood loss!
    def loss(parameters, inpt, targets):
        W, b = climin.util.shaped_from_flat(parameters, tmpl)
        classifier.W.set_value(W, borrow=True)
        classifier.b.set_value(b, borrow=True)

        return cost({x: inpt, y: targets})

    def d_loss_wrt_pars(parameters, inpt, targets):
        W, b = climin.util.shaped_from_flat(parameters, tmpl)
        d_flat, (d_W, d_b) = climin.util.empty_with_views(tmpl)
        classifier.W.set_value(W, borrow=True)
        classifier.b.set_value(b, borrow=True)

        d_W[...] = g_W.eval({x: inpt, y: targets})
        d_b[...] = g_b.eval({x: inpt, y: targets})

        return d_flat

    # Parameter Template
    tmpl = [(784, 10), 10]

    # Create space for the flat parameters
    wrt = numpy.zeros(7850)
    climin.initialize.randomize_normal(wrt, 0, 0.01)

    # datastream
    args = ((i, {}) for i in climin.util.iter_minibatches(
        [train_set_x, train_set_y], batch_size, [0, 0]))

    # build optimizer
    if optimizer == 'gradient_descent':
        print("[*] Using gradient descent ...")
        opt = climin.GradientDescent(wrt, d_loss_wrt_pars, step_rate=0.13, momentum=0, args=args)
    elif optimizer == 'rmsprop':
        print("[*] Using gradient descent ...")
        opt = climin.
    elif optimizer == 'adadelta':
        print("[*] Using adadelte ...")
        opt = climin.
    elif optimizer == 'adam':
        print("[*] Using adam ...")
        opt = climin.
    elif optimizer == 'reslilient_propagation':
        print("[*] Using resilient propagation ...")
        opt = climin.
    elif optimizer == 'conjugate_gradients':
        print("[*] Using conjugate gradients ...")
        opt = climin.
    elif optimizer == 'quasi_newton':
        print("[*] Using quasi newton ...")
        opt = climin.

    ###############
    # TRAIN MODEL #
    ###############
    print("[*] Training the model ...")

    # early stopping parameters
    patience = 5000  # look at minimum this number of examples
    patience_increase = 2  # wait this much longer when a new bet is founc
    improvement_threshold = 0.995  # relative imporvement that is considered significant
    validation_frequency = min(n_train_batches, patience // 2)  # check performance on validation set
                                                                # after this many minibatches
                                                                # here after 1 epoch

    best_validation_loss = numpy.inf
    test_loss = numpy.inf
    start_time = timeit.default_timer()

    for info in opt:
        if info['n_iter'] >= 1000:
            print test_model()
            break

    done_looping = False
    for info in opt:

        iteration = info['n_iter']
        epoch = iteration // n_train_batches
        minibatch_index = iteration % n_train_batches

        if iteration % validation_frequency == 0:
            # compute zero-one loss on validation set
            validation_loss = validate_model()

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
                    patience = max(patience, iter * patience_increase)

                best_validation_loss = validation_loss

                # test it on the test set
                test_loss = test_model()

                print(
                    "epoch %i, minibatch %i/%i, test error of best model %f %%" %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        test_loss * 100.
                    )
                )

                # save the best model
                with open('best_model.pkl', 'wb') as f:
                    pickle.dump(classifier, f)

        if patience <= iteration or epoch >= n_epochs:
            # set parameters in the LogReg class to make sure they are up to date
            set_pars(wrt)
            break

    end_time = timeit.default_timer()
    print(
        "[*] Optimization complete with best validation score of %f %%, with test performance %f %%" %
        (
            best_validation_loss * 100.,
            test_loss * 100.
        )
    )
    print("The code run for %d epochs, with %f epochs/sec" % (epoch, 1. * epoch / (end_time - start_time)))
    print(
    ("The code for file " + os.path.split(__file__)[1] + " ran for %.1fs" % ((end_time - start_time))), file=sys.stderr)

    with open('best_model.pkl', 'wb') as f:
        pickle.dump(classifier, f)

if __name__ == '__main__':
    opt()