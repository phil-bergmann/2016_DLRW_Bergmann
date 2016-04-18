import numpy
import theano
import theano.tensor as T
import six.moves.cPickle as pickle
import os
import gzip


class LogReg():
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

    def negativ_llh(self, y):
        """ Returns the negative log-likelihood

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """

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

def load_data(dataset):
    """ Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (MNIST)
    """

    #############
    # LOAD DATA #
    #############

    # Check if MNIST dataset is present, if not load it
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # check data directory
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://deeplearning.net/data/mnist/mnist.pkl.gz'
        )
        print('[*] Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('[*] Loading data ...')

    # load now
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple (input, target)
    # input is a numpy.ndarray with 2 dimensions, each row one example
    # traget is a numpy.ndarray with 1 dimension, same length as number
    # of rows from input. Gives the target to the corresponding example

    def shared_dataset(data_xy, borrow=True):
        """ Loads the dataset into shared variables

        This is mainly done to increase GPU performance. This allows theano to
        copy the data into the GPU memory.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # Data stored in the GPU has to be float, that's the reason we use
        # 'floatX'. But during computation we need them as ints and therefore
        # we need to cast them, before returning them. This is a nice little hack
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval