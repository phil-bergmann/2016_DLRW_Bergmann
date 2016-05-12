Depends on content of ../data and the whole content of this folder
Note: bh_tsne was compiled for Mac OS X, to run under windows compile and place at right place (I think ./windows/)

Problem 27:
check

Problem 28:
run_bhtsne.py
check, here one example of experiments done:
Run to try out different values of theta on a 10000 pictures subset of the grayscale cifar dataset. Outputs will be
saved to 'cifar_thetaX.png'. Run times were:

theta       Time(s)
0.5         139s
0.75        101s
1.0         79s

As can be seen choosing bigger thetas reduces the runtime, but if we look at the picture produced at 1.0 we see that
performance also decreases.

Note on parameters: inverse tells if all picture points should be inverted (255->0, 0->255). This only affects how the
pictures of the samples are displayed in the plot. Shape must be the shape of the plotted pictures.

Problem 29:
execute bhtsne_mnist.py
Outputs the picture to 'bhtsne_mnist.png'