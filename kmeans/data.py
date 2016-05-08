import six.moves.cPickle as pickle
import os
import gzip
import numpy
import theano
import theano.tensor as T
from PIL import Image

def load_data(dataset, shared=True):
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

    def not_shared_dataset(data_xy):
        data_x, data_y = data_xy
        data_x = numpy.asarray(data_x, dtype=theano.config.floatX)
        data_y = numpy.asarray(data_y, dtype='int32')

        return data_x, data_y

    if shared:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)
    else:
        test_set_x, test_set_y = not_shared_dataset(test_set)
        valid_set_x, valid_set_y = not_shared_dataset(valid_set)
        train_set_x, train_set_y = not_shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

def load_cifar(dataset='../data/cifar-10/', concat=True, label_names=False, small=False):
    names = ['data_batch_1_gray', 'data_batch_2_gray', 'data_batch_3_gray', 'data_batch_4_gray', 'data_batch_5_gray',
             'test_batch_gray']

    rval = []

    print('[*] Loading data ...')

    for n in names:
        fo = open(dataset+n, 'rb')
        dictionary = pickle.load(fo)
        fo.close()
        data = dictionary["data"]
        labels = dictionary["labels"]
        if small:
            data_tmp = numpy.empty((data.shape[0], 144))
            for i in xrange(data.shape[0]):
                image = Image.fromarray(data[i].reshape(32, 32))
                image.thumbnail((12, 12))
                data_tmp[i] = numpy.array(image.getdata(), dtype='uint8')
            data = data_tmp
        if len(rval) == 0 or not concat or n == 'test_batch_gray':
            rval.append((data, labels))
        else:
            rval[0] = (numpy.concatenate((rval[0][0], data)), numpy.concatenate((rval[0][1], labels)))

    if label_names:
        fo = open(dataset + 'batches.meta', 'rb')
        dictionary = pickle.load(fo)
        fo.close()
        labels = dictionary["label_names"]
        rval.append(labels)

    return rval
