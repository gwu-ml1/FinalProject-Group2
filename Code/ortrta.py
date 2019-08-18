from common_utils import load_cleaned, data_folder
from os import path
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model.logistic import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.metrics import cohen_kappa_score, \
    make_scorer, classification_report, accuracy_score
from joblib import load, dump
import pandas as pd

# Now I'm attempting to find 1 'ring' to rule them all.
#  by binding them... in the darkness... lol
def filepath(filename):
    return path.join(data_folder, 'persisted_models', filename)


def constructFeatures(X_train, X_test):
    # load pre-trained models
    classifiers = [
        ('bayes', load(filepath('bernoulliBayes.joblib'))),
        ('knn', load(filepath('knn.joblib'))),
        ('logReg', load(filepath('logReg.joblib'))),
        ('ranFor', load(filepath('ranFor.joblib'))),
        ('sklearn_mlp', load(filepath('sklearn_mlp.joblib'))),
        ('svm', load(filepath('svm.joblib')))
    ]

    X2_train = pd.DataFrame()
    X2_test = pd.DataFrame()

    for name, clf in classifiers:
        X2_train[name] = clf.predict(X_train)
        X2_test[name] = clf.predict(X_test)

    return X2_train, X2_test

myscorer = make_scorer(cohen_kappa_score)

def randomForest():
    return GridSearchCV(
        estimator=RandomForestClassifier(
            n_jobs=-1,
            random_state=42
        ),
        param_grid={
            'n_estimators': [5, 10, 20, 40, 80],
            'max_depth': [5, 25, 100, 200, None]
        },
        scoring=myscorer,
        cv=5
    )

def logReg():
    return LogisticRegressionCV(
        max_iter=200,  # doubled default max_iter value to help with convergence
        tol=0.01,  # increase tolerance above default by factor of 10 to help
                   #   with convergence.
        cv=5,  # Increased number of folds to 5 (default is 3)
        penalty='l1',  # Using l1 regularization as feature selection is implied
        solver='saga',  # Using saga solver as it supports l1 reg., and warm start
                       #   according to docs, warm start helps speed up CV loop
        random_state=42,
        scoring=myscorer,
        n_jobs=-1
    )

def knn():
    return GridSearchCV(
        estimator=KNeighborsClassifier(
            n_jobs=-1
        ),
        param_grid={
            'n_neighbors': [1,2,4,8,16]
        },
        n_jobs=1,
        cv=5,
        scoring=myscorer
    )

def bayes():
    return BernoulliNB()

def svc():
    return GridSearchCV(
        estimator=SVC(
            gamma='scale',
            random_state=42
        ),
        param_grid={
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        },
        n_jobs=-1,
        cv=5,
        scoring=myscorer
    )

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_cleaned()

    X2_train, X2_test = constructFeatures(X_train, X_test)

    models = [
        ('ranFor', randomForest()),
        ('logReg', logReg()),
        ('knn', knn()),
        ('bernoulliBayes', bayes()),
        ('svm', svc())
    ]

    for model_name, clf in models:
        print("\n\nBeginning to Train Model:", model_name)
        clf.fit(X2_train, y_train)
        y_predicted = clf.predict(X2_test)
        print("Model Training Complete:", model_name)
        print(classification_report(y_test, y_predicted))
        print("\tCohen's Kappa:", cohen_kappa_score(y_test, y_predicted))
        print("\tAccuracy:", accuracy_score(y_test, y_predicted))
        print("Persisting model to disk:", model_name)
        dump(clf, path.join(data_folder,
                            'persisted_models',
                            'stacked',
                            model_name + '.joblib'))
        print("Model Persisting Complete:", model_name)


