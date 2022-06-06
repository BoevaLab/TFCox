# TFCox

## Introduction
This is a Python implementation of the Cox Proportional Hazards model in Tensorflow and Keras. It is designed to be run on small p>>n datasets which other implementations are not optimised for. It inlcudes quality of life features such as early stopping at a set concordance threshold, optional normalization, and the ability to revert to the prior training epoch if infinite weights are obtained. The model uses the FTRL proximal gradient descent optimization method to solve the CoxPH model.

TFCox is designed with the sklearn style in mind and so should be straightforward to use for those familiar with sklearn.


## Cox Proportional Hazards Model

The Cox Proportional Hazards model is designed to evaluate the relative likelihood of the occurance of an event. It works on censored data where the event has occured for some samples but not others.

A detailed explanation of Proportional Hazards Models, including Cox, is available here:

https://en.wikipedia.org/wiki/Proportional_hazards_model

## Installation
The TFCox model file 'TFCox.py' contains the entire model class. There is no installation, merely download and include the file and import it directly. Or alternatively copy the class into your own code. It should work on Windows, Mac and Linux provided the underlying packages have been installed.

The packages required to use the model are as follows:

Tensorflow 2 (v2.3 onwards should work)
numpy (v1.20 onwards should work)
pandas (v1.2 onwards should work)

Note: This model requires TensorFlow eager to be turned off, the code will do this automatically, so if you need TF eager you will need to turn it back on yourself.
Note: The model was designed with tensorflow operating in gpu mode but should work in cpu only mode.
Note: Due to how the Cox model operates this does not train using batches, therefore large models are not recommended due to high memory requirements. Modification for batch training is possible but not recommended for convex optimization problems such as the CoxPH model.


## Usage

class TFCox(seed=42,norm=False,optimizer='Ftrl',l1_ratio=1,lbda=0.0001,
                 max_it=150,learn_rate=0.001,momentum=0.1,stop_if_nan=True,stop_at_value=False, 
                 cscore_metric=False,suppress_warnings=True,verbose=0)

### Parameters

- seed (int, default:42) : sets the random seed for the weight intialization.
- batch_norm (bool, default:False) : If True features will be normalized during training.
- optimizer (Ftrl, Adam, SGD, SGDmomentum, RMSprop, default=Ftrl) : This is the optimizer that is used with the model. It is strongly recommended to use the Ftrl optimizer since it converges better on convex problems such as CoxPH. As an alternative Adam will converge faster but conflics with regularization and will not necessarily converge to the correct solution. The other 3 options are not recommended but are provided as options for those who want them.
- l1_ratio (float range(0-1), default:1) : Controls the proportion of L1 and L2 regularization. l1_ratio = 1 is equivalent to LASSO and l1_ratio=0 is equivalent to Ridge.
- lbda (float, default=0.0001) : The regularization constant, this can be set to 0.
- max_it (int, default=150) : This is the maximum number of iterations before the model stops. default of 150 is used for Ftrl optimizer, for Adam optimizer 50 is a better choice.
- learn_rate (float, default=0.001) : This is the learning rate of the model. The default of 0.001 is good for Ftrl and Adam optimizers, should be increased if the model is used with gradient descent.
- momentum (float, default 0.1) : This is only used with the SGDmomentum optimizer, otherwise this does nothing.
- stop_if_nan (bool, default=True) : This stops the model if nan or infinite weights are reached and then returns the non nan weights from the previous epoch. NaN weights are generally very likely to occur, the two scenarios where it may happen are when the solution is undefined or when the learning rate is substantially increased.
- stop_at_value(False or float range(0-1), defaults=False) : This stops the model as soon as a target training concordance is reached. Note: Dramatically slows the model on larger datasets due to calculating concordance at each training epoch.
- cscore_metric (bool, default: False) : Calculates the concordance at each training epoch and saves it in loss_history_. Note: Dramatically slows the model on larger datasets due to calculating concordance at each training epoch.
- suppress_warnings (bool, default: True) : Suppresses tensorflow deprecation warnings. This is useful to avoid the dsiplay of large numbers of deprectaion warnings related to diabling eager mode. However, it is important to note that this will disable other potentially important warnings, so if anthing isn't working this should be disabled.
- verbose (0-1, default=0) : If verbose = 1 then loss at each epoch and the model summary will be printed during fit. Otherwise no information will be printed.

### Methods

**fit(X,y1,y2)**

**Parameters**
- **X**  (array like, shape = (n_samples,n_features)) - Data Matrix
- **y1** (array like, shape = (n_samples,1)) - Array of censoring states
- **y2** (array like, shape = (n_samples,1)) - Array of times

This is the fit method, it takes the X data and the state and time and computes the CoxPH model.


**predict(X)**

**Parameters**
- **X**  (array like, shape = (n_samples,n_features)) - Data Matrix

This is the predict method, it takes an array of data and returns the predicted hazard scores.

### Variables

- .beta_  gives the weight array of the model
- .loss_history_  gives the training loss and concordance after each training epoch.


## Example usage

Both an example on simulated data and the code to run a nested shuffle split on TCGA data are included. 
The required datasets can be found here: https://cloud.inf.ethz.ch/s/ssmgyQ8Y6fLz44A
