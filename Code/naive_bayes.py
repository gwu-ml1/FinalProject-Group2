from common_utils import load_cleaned
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.metrics import cohen_kappa_score, accuracy_score

#TODO: This isn't working yet...
#  latest error... X must be non-negative... ?
def naiveBayes(X_train, y_train):

    # Code below heavily inspired by sklearn documentation, in particular:
    # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#sphx-glr-auto-examples-compose-plot-column-transformer-mixed-types-py
    # building ingest pipeline using sklearn
    numeric_features = ['date', 'innings1_runs', 'innings1_wickets', 'innings1_overs', 'innings1_balls_bowled']

    categorical_features = ['venue', 'round', 'home', 'away', 'innings1', 'innings2']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Notice we're dropping numeric features entirely... I've doubts that this will perform well.
    # TODO: is there a mechanism for combining a GaussianNB on numberic features, and BernoulliNB on
    #   categorical one-hot encoded? sklearn does have some sort of ensemble models... interesting.
    #   as currently setup we get about .57 accuracy, and cohen's kappa of .16... so barely better
    #   than NullAccuracy rate ~52%. May imply that the categorical variables aren't a widely useful
    #   contributor.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num',  'drop', numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    nbmodel = BernoulliNB()

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        #  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html
        ('classifier', nbmodel)
    ])

    # this takes a few minutes to run...
    clf.fit(X_train, y_train)
    return clf, nbmodel

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_cleaned()
    pipe, model = naiveBayes(X_train, y_train)
    y_predict = pipe.predict(X_test)
    print("finished training Naive Bayes Model")
    print("\naccuracy:", accuracy_score(y_predict, y_test))
    print("\ncohen's kappa:", cohen_kappa_score(y_predict, y_test))