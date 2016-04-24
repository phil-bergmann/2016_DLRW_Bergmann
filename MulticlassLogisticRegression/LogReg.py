import numpy
import theano
import theano.tensor as T


class LogReg:
    """ Logistic Regression Class

    Oriented by http://deeplearning.net/tutorial/logreg.html
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the class

        :type input: theano.tensor.TensorType
        :type n_in: int
        :type n_out: int
        """

        # weight matrix
        self.W = theano.shared(
            value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
            name="W",
            borrow=True
        )

        # biases
        self.b = theano.shared(
            value=numpy.zeros((n_out), dtype=theano.config.floatX),
            name="b",
            borrow=True
        )

        # compute class membership probabilities
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute most probable class
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters
        self.params = [self.W, self.b]

        # input
        self.input = input

    def negative_llh(self, y):
        """ Returns the negative log-likelihood

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """ Returns a float representing the number of error in the minibatch over
        the total number of examples in the minibatch; zero one loss

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check dimensions
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )

        # check datatype of y
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
