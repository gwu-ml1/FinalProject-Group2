import pandas as pd
from common_utils import load_cleaned
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import cohen_kappa_score, make_scorer, accuracy_score

def knnClassifier(X_train, y_train):

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

    knnModel = GridSearchCV(
        estimator=KNeighborsClassifier(
            n_jobs=-1
        ),
        param_grid={
            'n_neighbors': [2, 3, 5, 7]
        },
        n_jobs=-1,
        cv=5,
        scoring=make_scorer(cohen_kappa_score)
    )

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        #  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html
        ('classifier', knnModel)
    ])

    # this takes a few minutes to run...
    clf.fit(X_train, y_train)
    return clf, knnModel

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_cleaned()
    pipe, model = knnClassifier(X_train, y_train)
    y_predict = pipe.predict(X_test)
    print("finished training Support Vector Classifier")
    print("\naccuracy:", accuracy_score(y_predict, y_test))
    print("\ncohen's kappa:", cohen_kappa_score(y_predict, y_test))