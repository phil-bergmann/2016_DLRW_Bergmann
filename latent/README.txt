Depends on content of ../data and the whole content of this folder

Problem 22+23:
autoencoder.py
Implementation of a autoencoder class. Optimizers included are gradient descent and rmsprop. For a example how to run
look at the end of the file (the file can directly be run). The initialization only requires the path to the dataset and
the number of the hidden units. All other parameters can be adjusted through the train() function. This has the advantage
that for multiple runs the model has to be built only one time and all parameters are stored in the class until the next
time update() is run.
test_params.py tries out different parameters for the number of neurons and the parameter lambda. The resulats are as
follows:
Neurons:    500     784     1000    1500    2000
lambda
    0       1.30    0.880   0.789   0.794   0.826
    0.001   1.51    1.19    1.16    1.22    1.27
    0.01    3.20    3.02    2.99    2.95    2.94
    0.1     9.66    9.39    9.30    9.29    9.45
    1.0     24.1    24.2    24.3    24.5    24.7

Looking at these numbers for future we see a slightly better cost for all lambdas for increasing number of neurons, which
seems plausible, because the encoder now can use more features to reconstruct the original image. But for bigger numbers
than 1000 neurons we can observe a kind of saturation and the costs become slightly worse. So for future tasks I will use
1000 neurons, which seem to be a good compromise for all values of lambda.

Problem 24 + 25:
receptive_reconstruct.py
Just run the file and it will output 'autoencoderrec_lambdaX.png' and 'autoencoderfilter_lambdaX.png'. It will be
run with 1000 neurons and different values for lambda (X)

Problem 26:
A sparse encoding of the MNIST dataset is a compressed representation of the different test samples in the dataset. Given
only the hidden unit activations of a test one can reconstruct the the original image with the upper part of the
autoncoder. So the sparse encoding captures the most important features of the whole dataset and ignores not so important
ones. Given that, a sparse encoding is a very good pretraining for a multilayer neural network and yields much faster
convergence and better results when used.

Bonus problem:
set kl (beta) and p parameters in the train function to activate kullback leiber divergence. Just run autoencoder_kl.py.
Note: It was pretty slow, around a few seconds per minibatch, so I could not test the KL-Divergence. I also created a second
file for the kl divergence implementation, beacause it also slows down computation, if the kl parameter is set to 0.
