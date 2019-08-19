from common_utils import load_cleaned
from logistic_regression import logRegModel
from random_forest import randomForestModel
from knn_classifier import knnClassifier
from svm_classifier import svm
from naive_bayes import naiveBayes
from sklearn_mlp import mlp as sklearn_mlp
from keras_mlp import mlp as keras_mlp
from joblib import dump
from os import path

from sklearn.metrics import classification_report, accuracy_score, cohen_kappa_score

data_dir = '../Data'

def train_and_persist(X_train, y_train, X_test, y_test,
                      model_builder, model_name):
    print("\n\nBeginning to Train Model:", model_name)
    pipe, model = model_builder(X_train, y_train)
    y_predicted = pipe.predict(X_test)
    print("Model Training Complete:", model_name)
    print(classification_report(y_test, y_predicted))
    print("\tAccuracy:", accuracy_score(y_test, y_predicted))
    print("\tCohen Kappa:", cohen_kappa_score(y_test, y_predicted))
    print("Persisting model to disk:", model_name)
    dump(pipe, path.join(data_dir, 'persisted_models', model_name + '.joblib'))
    print("Model Persisting Complete:", model_name)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_cleaned(data_dir)

    # compute null accuracy here for comparison purposes
    #   this way of computing null accuracy was found here:
    #   https://www.ritchieng.com/machine-learning-evaluate-classification-model/
    print("test set null accuracy:",
          max(y_test.mean(), 1 - y_test.mean()))

    models = {
        'logReg': logRegModel,
        'ranFor': randomForestModel,
        'svm': svm,
        'knn': knnClassifier,
        'bernoulliBayes': naiveBayes,
        'sklearn_mlp': sklearn_mlp  #,
        #  did this separately as it takes longer to train.
        #'keras_mlp': keras_mlp
    }

    for (model_name, model_builder) in models.items():
        train_and_persist(X_train, y_train, X_test, y_test,
                          model_builder, model_name)
