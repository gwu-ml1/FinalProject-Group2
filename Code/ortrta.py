from common_utils import load_cleaned, data_folder, load_trained_models
from os import path
from os.path import join
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model.logistic import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.metrics import cohen_kappa_score, \
    make_scorer, classification_report, accuracy_score, confusion_matrix, \
    roc_curve, auc
from joblib import dump
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import randint as sp_randint


# Now I'm attempting to find 1 'ring' to rule them all.
#  by binding them... in the darkness... lol

def constructFeatures(X_train, X_test):
    # load pre-trained models
    classifiers = load_trained_models()

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
    return RandomizedSearchCV(
        estimator=KNeighborsClassifier(
            n_jobs=-1
        ),
        param_distributions={
            'n_neighbors': sp_randint(1, 800)
        },
        n_iter=50,
        n_jobs=1,
        cv=5,
        scoring=myscorer,
        random_state=42
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
        print(confusion_matrix(y_test, y_predicted))
        if model_name in ['svm']:
            y_predict_all = clf.decision_function(X2_test)
        else:
            y_predict_all = clf.predict_proba(X2_test)[:, 1]
        plt.figure()
        plt.title('Stacked ' + model_name + " ROC Curve")
        fpr, tpr, _ = roc_curve(
            y_test,
            y_predict_all)

        fpr1, tpr1, _ = roc_curve(
            y_test,
            y_predicted)

        # print(thresholds)
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label='ROC curve (area = %0.2f)' %
                       auc(fpr, tpr))
        plt.plot(fpr1, tpr1, color='cyan', lw=2,
                 label='ROC curve (area = %0.2f)' %
                       auc(fpr1, tpr1))
        plt.plot([0, 1], [0, 1], color='navy', lw=2,
                 linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.savefig(join(data_folder, 'images', 'stacked_' + model_name + '.png'))
        plt.show()



