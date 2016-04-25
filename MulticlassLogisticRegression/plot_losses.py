from __future__ import print_function
import matplotlib.pyplot as plt
from logistic_optimizer import opt

optimizers = ['gradient_descent', 'rmsprop', 'adadelta', 'adam', 'resilient_propagation',
              'nonlinear_conjugate_gradients', 'quasi_newton_lbfgs']


figure, (axes) = plt.subplots(7, 1)

for i in range(len(optimizers)):
    tr_loss, va_loss, te_loss, x_axis = opt(optimizer=optimizers[i], verbose=True)
    axes[i].set_title(optimizers[i])
    tr = axes[i].plot(x_axis, tr_loss, label="train error")
    va = axes[i].plot(x_axis, va_loss, label="validation error")
    te = axes[i].plot(x_axis, te_loss, label="test error")


axes[-1].legend(loc=0, shadow=True, fontsize='x-small')

figure.set_size_inches(15, 10)
figure.subplots_adjust(hspace=0.5)
figure.savefig('error.png')

plt.close(figure)


