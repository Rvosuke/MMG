# Bayesian Optimization

## probability model of the objective function

we use it to evaluate in the true objective function.

Let's say we just have 10 samples from the true objective function.

![img](https://miro.medium.com/v2/resize:fit:1033/1*OlgnEpytSBp464iWR9y_qQ.png)

## surrogate model (response surface model)

approximate the true objective model.

![img](https://miro.medium.com/v2/resize:fit:607/1*DEvsSJ1qW3NxePKuHciXqA.png)

the surrogate model is essentially a model trained on the $(hyperparameter,\text{true objective function score})$ pairs.

In math, it is $P(\text{objective function score},hyperparameter)$.

## acquisition function (selection function)

the next hyperparameter of choice is where the acquisition function is maximized.

The green shade is the acquisition function and the red straight line is where it is maximized.

![img](https://miro.medium.com/v2/resize:fit:611/1*9EszMI-ff2PbEPl38LpMQw.png)

## repeat

After using the acquisition function to determine the next hyperparameter, the true objective function score of this new hyperparameter is obtained. Since the surrogate model has trained on the $(hyperparameter,\text{true objective function score})$ pairs, adding a new data points updates the surrogate model.

![img](https://miro.medium.com/v2/resize:fit:1220/1*Da6iVML97uKuekBehfrUKw.png)