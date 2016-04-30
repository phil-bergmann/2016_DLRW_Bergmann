import matplotlib.pyplot as plt
import theano.tensor as T

from test_mlp import test_mlp

def plot():
    tanh = test_mlp(n_epochs=1000, batch_size=20, n_hidden=300, optimizer='gradient_descent', verbose=True,
                    activation=T.tanh, plot="repfield_tanh.png", hidden_name="TanH")
    sig = test_mlp(n_epochs=1000, batch_size=20, n_hidden=300, optimizer='gradient_descent', verbose=True,
                    activation=T.nnet.sigmoid, plot="repfield_sigmoid.png", hidden_name="Sigmoid")
    relu = test_mlp(n_epochs=1000, batch_size=20, n_hidden=300, optimizer='gradient_descent', verbose=True,
                    activation=T.nnet.relu, plot="repfield_relu.png", hidden_name="Relu")

    losses = [tanh, sig, relu]
    titles = ["Tangens Hyperbolicus", "Sigmoid", "Rectified Linear Unit"]

    figure, (axes) = plt.subplots(3, 1)

    for i in range(3):
        tr_loss, va_loss, te_loss, x_axis = losses[i]
        axes[i].set_title(titles[i])
        tr = axes[i].plot(x_axis, tr_loss, label="train error")
        va = axes[i].plot(x_axis, va_loss, label="validation error")
        te = axes[i].plot(x_axis, te_loss, label="test error")

    axes[-1].legend(loc=0, shadow=True, fontsize='x-small')

    #figure.set_size_inches(15, 10)
    figure.subplots_adjust(hspace=0.5)
    figure.savefig('error.png')
    plt.show()

    plt.close(figure)

if __name__ == '__main__':
    plot()