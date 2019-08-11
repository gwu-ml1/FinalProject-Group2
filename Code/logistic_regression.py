import pandas as pd
from common_utils import load_cleaned
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model.logistic import LogisticRegressionCV
from sklearn.metrics import cohen_kappa_score, accuracy_score, make_scorer

def logRegModel(X_train, y_train):

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

    logRegModel = LogisticRegressionCV(
        max_iter=200,  # doubled default max_iter value to help with convergence
        tol=0.01,  # increase tolerance above default by factor of 10 to help
                   #   with convergence.
        cv=5,  # Increased number of folds to 5 (default is 3)
        penalty='l1',  # Using l1 regularization as feature selection is implied
        solver='saga',  # Using saga solver as it supports l1 reg., and warm start
                       #   according to docs, warm start helps speed up CV loop
        random_state=42,
        scoring=make_scorer(cohen_kappa_score)
    )

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        #  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html
        ('classifier', logRegModel)
    ])

    # this takes a few minutes to run...
    clf.fit(X_train, y_train)
    return clf, logRegModel

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_cleaned()
    pipe, model = logRegModel(X_train, y_train)
    y_predict = pipe.predict(X_test)
    print("finished training Logistic Regression Model")
    print("\taccuracy:", accuracy_score(y_test, y_predict))
    print("\tcohen's kappa:", cohen_kappa_score(y_test, y_predict))
