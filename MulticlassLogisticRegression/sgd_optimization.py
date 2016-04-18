from __future__ import print_function

import numpy
import theano
import theano.tensor as T
import six.moves.cPickle as pickle
import timeit
import os
import sys

from LogReg import LogReg
from LogReg import load_data




def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000, dataset='mnist.pkl.gz', batch_size=600):
    """ Stochastic gradient descent of a log-linear model

    Demonstration on MNIST dataset.

    :type learning_rate: float
    :param learning_rate: learning rate used for stochastic gradient

    :type n_epochs: int
    :param n_epochs: maximal number of epochs run

    :type dataset: string
    :param dataset: the path to the MNIST dataset file

    :type batch_size: int
    :param batch_size: size of the batches used for gradient descent
    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute the number of minibatches, be aware ignors a few examples if divison is with remainder
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print("[*] Building model ...")

    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')  # data
    y = T.ivector('y')  # class labels

    # build logistic regression classifier
    # Each MNIST image has size 28*28
    classifier = LogReg(input=x, n_in=28*28, n_out=10)

    # cost to minimize is negative log-likelihood
    cost = classifier.negativ_llh(y)

    # compile functions that messure the mistakes made by the model on each minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # define how to update the parameters of the model
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
           (classifier.b, classifier.b - learning_rate * g_b)]

    # compile a function to train the model with rules in 'updates,
    # which also returns the cost
    train_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
        }
    )

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
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    "epoch %i, minibatch %i/%i, validation error %f %%" %
                      (
                          epoch,
                          minibatch_index + 1,
                          n_train_batches,
                          this_validation_loss * 100.
                      )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss

                    # test it on the test set
                    test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        "epoch %i, minibatch %i/%i, test error of best model %f %%" %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)
            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        "[*] Optimization complete with best validation score of %f %%, with test performance %f %%" %
        (
            best_validation_loss * 100.,
            test_score * 100.
        )
    )
    print("The code run for %d epochs, with %f epochs/sec" % (epoch, 1.*epoch / (end_time - start_time)))
    print(("The code for file " + os.path.split(__file__)[1] + " ran for %.1fs" % ((end_time - start_time))), file=sys.stderr)


def predict():
    """ Example function to predict labels

    """

    # load the saved model
    classifier = pickle.load(open('best_model.pkl'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    # We can test it on some examples from test set
    dataset = 'mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)

if __name__ == '__main__':
    sgd_optimization_mnist()