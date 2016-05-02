from __future__ import print_function

import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import timeit
import os
import climin
import climin.util
import sys

from dA import dA
from data import load_data

class autoencoder():

    def __init__(self, dataset='mnist.pkl.gz', n_hidden=500):
        datasets = load_data(dataset)

        self.train_set_x, train_set_y = datasets[0]
        #valid_set_x, valid_set_y = datasets[1]
        self.test_set_x, test_set_y = datasets[2]

        # W, b, b_prime
        self.template = [(28 * 28, n_hidden), n_hidden, 28 * 28]

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images

        #####################################
        # BUILDING THE MODEL CORRUPTION 30% #
        #####################################

        rng = numpy.random.RandomState(123)
        theano_rng = RandomStreams(rng.randint(2 ** 30))

        self.da = dA(
            numpy_rng=rng,
            theano_rng=theano_rng,
            input=self.x,
            n_visible=28 * 28,
            n_hidden=n_hidden
        )

        ##########################
        # computing the gradient #
        ##########################

        y = self.da.get_hidden_values(self.x)
        z = self.da.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        #L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        L = (z-self.x)**2
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.da.params)

        self.grad = theano.function(
            inputs=[self.x],
            outputs=gparams
        )

        self.train_cost = theano.function(
            inputs=[],
            outputs=cost,
            givens={
                self.x: self.train_set_x
            }
        )

    def initialize(self):
        # CLIMIN
        # wrt = numpy.zeros(28 * 28 * n_hidden + n_hidden + 28*28, dtype=theano.config.floatX)
        self.wrt, (self.params) = climin.util.empty_with_views(self.template)

    def set_pars(self):
        for p in zip(self.da.params, self.params):
            p[0].set_value(p[1], borrow=True)

    def train(self, learning_rate=0.1, training_epochs=15, batch_size=20, optimizier="g_d", corruption=0):
        # initialize weights
        self.initialize()

        # data
        args = ((i, {}) for i in climin.util.iter_minibatches(
            [self.train_set_x.eval()], batch_size, [0, 0]))

        # minibatches per epoch
        n_train_batches = self.test_set_x.get_value().shape[0] // batch_size

        # set corruption
        self.corruption = corruption

        def d_loss_wrt_pars(parameters, inpt):
            self.set_pars()
            d = [d for d in self.grad(inpt)]
            return numpy.concatenate((d[0].flatten(), d[1], d[2]))

        if optimizier == "g_d":
            print("[*] using GRADIENT DESCENT ...")
            opt = climin.GradientDescent(self.wrt, d_loss_wrt_pars, step_rate=learning_rate, momentum=0.0,
                                         momentum_type='nesterov', args=args)
        elif optimizier == "rmsprop":
            print("[*] using RMSPROP ...")
            # learning rate 0.0001
            opt = climin.RmsProp(self.wrt, d_loss_wrt_pars, step_rate=learning_rate, decay=0.9, momentum=0.0,
                                 step_adapt=False, step_rate_min=0, step_rate_max=numpy.inf, args=args)
        else:
            print("[*] no valid optimizer selected!")
            print("[*] shutting down ...")
            return 1

        ############
        # TRAINING #
        ############

        start_time = timeit.default_timer()
        # go through training epochs
        for info in opt:
            iteration = info['n_iter']
            epoch = iteration // n_train_batches
            # go through training set
            if iteration % n_train_batches == 0:
                print('Training epoch %d, cost ' % epoch, self.train_cost())

            if epoch >= training_epochs:
                self.set_pars()
                break

        end_time = timeit.default_timer()

        training_time = (end_time - start_time)

        print(('The 30% corruption code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.2fm' % (training_time / 60.)), file=sys.stderr)


if __name__ == '__main__':
    aec = autoencoder()
    aec.train()
