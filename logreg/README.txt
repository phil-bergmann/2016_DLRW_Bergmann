Depends on content of ../data and the whole content of this folder

Problem 8:
sgd_optimization.py

Problem 9:
Run sgd_optimization.py
Error rates for validation and trainig set go down to ~7.5%

Problem 10:
logistic_optimizer.py
Added support for different optimizers from the climin library and uses early stopping
The 'optimizer' argument specifies the optimizer to use. Best results were achieved with 'adam' and 'rmsprop'
For use just run the script.

Problem 11:
receptive_fields.py
Run to generate the 'repflds.png' image (already done, 'adam'). This implementation shows the receptive fields for saved
*.pkl models of the LogReg class and saves the images to 'repflds.png' (overrides previus ones). Run to see 'rmsprop'
and 'adam' receptive fields.

Problem 12:
Run plot_losses.py - Warning overrides original error.png!
Plots the losses for different optimizers into error.png
You should always stop training as soon as the training error keeps decreasing but the validation error doesn't for a long
time or even increases. This is the point where the model starts to overfit the training data and looses generalisation
capabilities.Determining how long to wait for further improvements of the validation error is the difficult thing.

Problem 13:
best.py
generate() - Tries to generate a good model, with a promising optimizer and early stopping is configured loose to not
stop too soon. Uses the 'adam' optimizer and saves model to 'adam.pkl'. Results were 6.69% error on the validation set
and 7.22% on the test set. Just run the script, it will execute validate() and test the model on the validation and test
set and output the results.

Bonus question: By enhancing the data set we create more data for out algorithm to perform better. While this would be
completely legal in a commercial product where only results count, it is bad in science. The MNIST data set was created
to be able to compare the performance of different classification algorithms and by enhancing the data our 'bad' algorithm
gains an unfair advantage. If now our algorithm performs better it's not because it is such a capable and good algorithm,
but simply because it has more data and now the sense of MNIST data set (a common set for comparing algorithms) is lost.