Depends on content of ../data and the whole content of this folder

Problem 14:
test_mlp.py
Run to test the neural network. The test_mlp() function accepts different parameters to adjust the net. Currently
gradient descent, rmsprop and rprop are supported, both taken from the climin library (adjust parameters if use of momentum is
desired). L1 and l2 regularisation are also implemented and can be activated by setting their parameter to >0.


Problem 15:
These test runs are too take a look at parameter settings, and stopped them at epoch 150 too be able to compare more
parameters in a reasonable amount of time. I know that parameters that initially converge fast can perform bad when the
weights are near the minimum, but here I can already see which parameters don't work at all.
Test runs (max epochs 150, early stopping, l2=0.0001)

Minibatches with 20 samples

optimizer   learning rate   momentum    type        valid   test    epoch
g_d         0.01            0.9         standard    1.67    1.98    143
g_d         0.001           0.9         standard    1.93    2.03    144
                                                    1.73    1.82    303
g_d         0.01            0.9         nesterov    1.65    1.97    37 - finished at 74
g_d         0.001           0.9         nesterov    1.95    2.12    147
g_d         0.1             0                       1.61    1.71    138
g_d         0.01            0                       1.97    2.05    137
g_d         0.001           0                       4.45    5.04    150

(slower)    step_rate       decay
rmsprop     0.0001          0.9                     1.86    1.97    141
rmsprop     0.001           0.9                     2.16    2.12    45 - finished 48

Full batch learning (set batch_size to 50000, max epochs to 2000)
rprop (standard parameters)                         1.75    1.76    1154 (55 epochs/min)

Problem 16:
First of all the input vectors should be normalized zero mean and unit variance. Because all inputs correspond to pixels
and have values between 0.0 and 1.0 they already have similar variance and probably their mean is at ~0.5. Of course the
pixels near to certain places (e.g. center) will be activated with a higher probability and thus move their mean value,
but overall the data is already in a pretty neural network friendly shape. The output should be in a good shape, so no
change needed here (as long as all number appear roughly the same amount of time). Now a simple change that would make
sense is to subtract 0.5 to move the mean more to 0. To do that set move_mean=-0.5. I did one test run with gradient
descent and learning rate 0.01 without momentum and also stopped at maximal 1500 epochs. Results:
                    validation error    test error  epoch
Without moving:     1.71                1.70        1090
With moving:        1.63                1.78        1023
It seems that moving the mean did only decrease the validation error, but increased the test error. So overall I see no
big positive effect and will just leave the data as it is.

Now let's look at the different activation functions and test if we can increase the learning speed with good weight
initialization. To limit the time needed for training I will only train until epoch 300, as we should already see there
if there has been a positive effect. The standard initialization method for all above examples were random samples drawn
from a gaussian distribution with zero mean and 0.01 standard deviation. First let's look at tanh hidden layer. The tutorial
cites that a good interval would be between +-sqrt(6/(fan_in+fan_out) drawn from an uniform distribution (+-0.14). Because
Climin doesn't offer a uniform initialization I will use a normal one. For sigmoidal hidden units the same value times 4
is used. The RELU units behave a little bit different, since they are zero for all inputs <0. So there exist two different
approaches, the first is to use an interval between sqrt(2/fan_in) and the second ist to use the standard initalization and add
a small positve constant to the bias, to ensure that the RELU units are in their working working range.
init method         validation error    test error  epoch

TANH
standard            1.80                1.79        292
enhanced            1.75                1.87        274

SIGMOID
standard            2.52                2.67        299
enhanced            2.57                2.68        288

RELU
standard            1.64                1.85        283
enhanced            1.73                1.81        274
bias shift (1)      1.90                2.01        242

To sum up the results, here nearly no effects of the usage of another initialization method could be observed. Maybe bigger
networks need to be constructed no observe bigger effects. At least none of the initialization methods worked particulary
bad and thus the network seems to be quite robust and converges to a small test/validation error no matter where it started

Problem 17:
-> See problem 18

Problem 18:
plot.py
Run the script, it will train 3 networks (tanh, sigmoid, relu), save their receptive fields as repfield_name.png and at
the end plot the error curves and save them in error.png.

Problem 19:
Already done, just look at the repfield outputs from problem 17, error rates are written on the repfield pngs.
