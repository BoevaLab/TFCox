# TFCox

## Introduction
This is a Python implementation of the Cox Proportional Hazards model in Tensorflow and Keras. It is designed to be run on small p>>n datasets which other implementations are not optimised for. It inlcudes quality of life features such as early stopping at a set concordance threshold, optional normalization, and the ability to revert to the prior training epoch if infinite weights are obtained. 

TFCox is designed with the sklearn style in mind and so should be straightforward to use for those familiar with sklearn.


## Cox Proportional Hazards Model

The Cox Proportional Hazards model is designed to evaluate the relative likelihood of the occurance of an event. It works on censored data where the event has occured for some samples but not others.

A detailed explanation of Proportional Hazards Models, including Cox, is available here:

https://en.wikipedia.org/wiki/Proportional_hazards_model

## Installation
The TFCox model file 'TFCox.py' contains the entire model class. There is no installation, merely download and include the file and import it directly. Or alternatively copy the class into your own code. It should work on Windows, Max and Linux provided the underlying packages have been installed.

The packages required to use the model are as follows:

Tensorflow 2 (v2.3 onwards should work)
numpy (v1.20 onwards should work)
pandas (v1.2 onwards should work)

Note: This model requires TensorFlow eager to be turned off, the code will do this automatically, so if you need TF eager you will need to turn it back on yourself.
Note: The model was designed with tensorflow operating in gpu mode but should work in cpu only mode.
Note: Due to how the Cox model operates this does not train using batches, therefore large models are not recommended due to high memory requirements.


## Usage

class TFCox(seed=42,batch_norm=False,l1_ratio=1,lbda=0.0001,
                 max_it=50,learn_rate=0.01,stop_if_nan=True,stop_at_value=False)


### Parameters

- seed (int, default:42) : sets the random seed for the weight intialization.
- batch_norm (bool, default:Falso) : If True features will be normalized during training.
- l1_ratio (float range(0-1), default:1) : Controls the proportion of L1 and L2 regularization. l1_ratio = 1 is equivalent to LASSO and l1_ratio=0 is equivalent to Ridge.
- lbda (float, default=0.0001) : The regularization constant, this can be set to 0, if so one of the stop flags should probably be turned on.
- stop_if_nan (bool, default=True) : This stops the model if nan or infinite weights are reached and then returns the non nan weights from the previous epoch.
- stop_at_values(False or float range(0-1), defaults=False) : This stops the model as soon as a target training concordance is reached.

### Methods

**fit(X,y1,y2)**

**Parameters**
- **X**  (array like, shape = (n_samples,n_features)) - Data Matrix
- **y1** (array like, shape = (n_samples,1)) - Array of censoring states
- **y2** (array like, shape = (n_samples,1)) - Array of times

This is the fit method, it takes the X data and the state and time and computes the CoxPH model.


**predict(X)

**Parameters
- **X**  (array like, shape = (n_samples,n_features)) - Data Matrix

This is the predict method, it takes an array of data and return the predicted hazard scores.

### Variables

- .beta_  gives the weight array of the model
- .model_history_  gives the training loss and concordance after each training epoch.


## Example usage

Both a basic example and the code to run a nested shuffle split on TCGA data are included.
