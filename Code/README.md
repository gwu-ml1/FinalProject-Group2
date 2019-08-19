# Code

This file attempts to describe the code written for the project

## Pre-Processing
### Data Ingestion

Our initial data pre-processing code is contained within the 
`data.py` file. The code reduces number of observations by about 
5 to 6 percent, due to some of the matches being abandoned.
We also reduce the number of features by about half, primarily
those dropped features are not of interest as they are only 
available after the match is over, and we're targeting predictions 
based on data available immediately following the 1st inning.

The code in `data.py` reads the `t20_matches.csv` file in the 
`Data` directory. It then reduces the dataset as described, and 
saves the resulting dataset in `cleaned.csv`.

### Loading Cleaned Dataset

For repeatably building multiple models, it seemed necessary to 
setup a function for loading the data in `cleaned.csv`, and 
appropriately setting the data types to be used for each feature. 
All the model training code subsequently used for our experiments
will use the `load_cleaned` function defined in `common_utils.py`.

## Experiments

In our attempt to find a model for predicting the outcome of the
matches, we looked at a number of different statistical models, and 
a Multi-Layer Perceptron based Neural Network. 

Each of the files setup with experiment contents, are setup the same
way. There is a function that build and trains the model, and a main
section towards the bottom of the the file for executing the function 
and printing accuracy and cohen kappa scores based on the test data. 
The experiments are constructed using the sci-kit learn Pipeline api. 
Where appropriate, we've utilized the `GridSearchCV`, and 
`RandomizedSearchCV` functions out of sci-kit learn to help identify 
appropriate values for hyperparameters. 

To collect up all the experiments in one script, I've added the code
in `train_and_persist.py`, which allows us to build and fit all the 
models based on the same initial train and test split, print the 
same classification report (as provided by sklearn, for each model)
and persist the finalized model to disk. 

*Except! The Keras based MLP Model, which I was unable to get to 
persist using the same joblib dump function used on the rest of 
the sklearn models. I'm sure there is a way to persist the keras 
model as well, but wasn't to worries about figuring it out.*

### Logistic Regression Classifier

The code for our logistic regression experiments are located in 
`logistic_regression.py`. sci-kit learn's apis provide a 
`LogisticRegressionCV` class, as it's optimized for cross validated
training and evaluation of logistic regression models. We 
chose, after some experimentation to utilize `l1` regularization, 
as it's capable of completely reducing the coefficients for features
to zero, implying that feature selection is a build-in aspect of 
the model when `l1` regularization is used. Logistic Regression 
was the first model we experimented with on the dataset, and also
proved to be one of the most successful. 

### Random Forest Classifier

The code for our Random Forest classification experiments are located
in `random_forest.py`. We utilized the `GridSearchCV` sklearn model 
to look for the best values for the maximum depth of the trees, and 
the number of trees to use in the random forest model. 

### KNearestNeighbors Classifier

The code for our KNearestNeighbors Classification experiments are 
located in `knn_classifier.py`. We utilized the `RandomizedSearchCV`
sklearn model to look for the best number of neighbors to use.

### Bernoulli Naive Bayes Classifier

The code our naive bayes classification experiments are located in 
`naive_bayes.py`. We were unable to figure out how to use the 
continuous features for naive bayes, instead settling on a bernoulli
specific version of naive bayes provided by the sklearn api. This 
model didn't perform particularly well relatively speaking, likely 
due to removing the continuous features from the dataset. 

### Support Vector Machine Classifier

The code our support vector machine classifier experiments are
located in `svm_classifier.py`. We used sklearn's `GridSearchCV` 
class to seek out the best kernel function based on trial and 
error. 

### Sklearn based Multi-layer Perceptron

The code for our sklearn based MLP experiments are located in
`sklearn_mlp.py`. We used sklearn's `GridSearchCV` and a lot of 
different searches to attempt to find reasonable hyperparameters. 
These included: 
 * activation function
 * solver algorithm
 * learning rate
 * number of neurons in the hidden layers
 * number of hidden layers
 * maximum number of iterations    

One major take away from this, was a new appreciation for the GPU
acceleration, available in keras, and the libraries backing keras. 
At some point, I decided training the hyperparameters using the 
sklearn was taking too long, and went with the best parameters
I'd been able to identify at that point. You'll notice in the code
that we only train one MLP model, based on prescribed hyperparameters. 
This is because it took a very long time to run GridSearchCV-like
operations on the sklearn based MLP model. 

### Keras based Multi-layer Perceptron

The code for our keras based MLP experiments are located in 
`keras_mlp.py`. We used sklearn's `RandomizedSearchCV` class to 
search for the best settings for several hyperparameters, though 
admittedly many others could have been added. 
Included Hyperparamerters:


