Depends on content of ../data

Problem 8:
sgd_optimization.py

Problem 9:
Run sgd_optimization()
Error rates for validation and trainig set go down to ~7.5%

Problem 10:
logistic_optimizer()
Added support for different optimizers from the climin library and uses early stopping
The 'optimizer' argument specifies the optimizer to use. Best results were achieved with 'adam' and 'rmsprop'
For use just run the script.

Problem 12:
Run plot_losses()
Plots the losses for different optimizers int error.png
You should always stop training as soon as the training error decreases but the validation error increases. This
is the point where the model starts to overfit the training data and looses generalisation capabilities.