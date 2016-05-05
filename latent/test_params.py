import numpy

from autoencoder import autoencoder

def test_params():

    neurons = [500, 784, 1000, 1500, 2000]

    costs = numpy.zeros((5, len(neurons)))

    for n in xrange(len(neurons)):
        aec = autoencoder(n_hidden=neurons[n])
        costs[0, n] = aec.train()
        costs[1, n] = aec.train(sparse=0.001)
        costs[2, n] = aec.train(sparse=0.01)
        costs[3, n] = aec.train(sparse=0.1)
        costs[4, n] = aec.train(sparse=1.0)

    print(costs)

if __name__ == '__main__':
    test_params()