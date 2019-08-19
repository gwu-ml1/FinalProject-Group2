import pandas as pd
from common_utils import load_cleaned
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import cohen_kappa_score, accuracy_score,\
    classification_report, make_scorer
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

# inspired by https://keras.io/getting-started/sequential-model-guide/
#   and https://keras.io/scikit-learn-api/
def construct_model(dropout_rate=0.0,
                    extra_layers=1,
                    num_neurons=455):
    # num inputs: 1867
    #   seems to fix a memory issue.... no idea
    backend.clear_session()

    model = Sequential()
    model.add(Dense(num_neurons,
                    activation='relu',
                    input_dim=1867)),
    for i in range(extra_layers):
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_neurons,
                        activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(2,
                    activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def mlp(X_train, y_train):

    # Code below heavily inspired by sklearn documentation, in particular:
    # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#sphx-glr-auto-examples-compose-plot-column-transformer-mixed-types-py
    # building ingest pipeline using sklearn
    #
    # using the docs as an example we'll setup a pipeline for scaling our numerical inputs, and onehotencoding our
    #   categorical inputs, then we'll tie this into a logistic regression model using an elasticnet penalty function.
    #   Ultimately this means we'll be able to figure out the appropriate hyperparameter p for the elasticnet penalty.
    numeric_features = ['date', 'innings1_runs', 'innings1_wickets', 'innings1_overs', 'innings1_balls_bowled']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), #TODO: median imputation make sense here?
        ('scaler', StandardScaler())])

    categorical_features = ['venue', 'round', 'home', 'away', 'innings1', 'innings2']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # number of features after pre-processing is complete: 1867
    #   attempt to build sequential model..
    # attempting to use randomized search b/c we can have the computer
    #   generate random numbers based on provided distribution functions
    #   which ultimately... is just an interesting thing. apparently
    #   gradient descent isn't such a great idea for parameter tuning...
    #   though the thought crossed my mind:
    #   https://stackoverflow.com/questions/43420493/sklearn-hyperparameter-tuning-by-gradient-descent
    # the randomized search stuff below, helped in part by:
    #   https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
    # setting up the GPU accelerated support from tensor-flow involved installing a number of libraries
    #   https://www.tensorflow.org/install/gpu
    #   Note: we did not install the optional 
    mlpModel = RandomizedSearchCV(
        estimator=KerasClassifier(
            build_fn=construct_model
        ),
        # comments capture best setting so far
        param_distributions={
            'epochs': sp_randint(1, 10),  # 3
            'dropout_rate': sp_uniform(0, 1),  # 0.8532567453865779
            'extra_layers': sp_randint(0, 15),  # 0
            'num_neurons': sp_randint(100, 960),  # 435
        },
        n_jobs=1,
        n_iter=10,
        cv=5,
        scoring=make_scorer(cohen_kappa_score),
        random_state=42
    )

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        #  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html
        ('classifier', mlpModel)
    ])

    # this takes a few minutes to run...
    clf.fit(X_train, y_train)
    return clf, mlpModel

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_cleaned()
    pipe, model = mlp(X_train, y_train)
    y_predict = pipe.predict(X_test)
    print("finished training Keras based Multi-layer perceptron")
    print(classification_report(y_test, y_predict))
    print("\taccuracy:", accuracy_score(y_test, y_predict))
    print("\tcohen's kappa:", cohen_kappa_score(y_test, y_predict))
    # manually connected a debugger python console in pycharm,
    #   imported dump from joblib, and dumped the model contents to
    #   disk.

