from common_utils import load_cleaned, load_trained_models, data_folder
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from os.path import join


X_train, X_test, y_train, y_test = load_cleaned()
combined = X_train
combined['target'] = y_train
# plt.hist(df.target.apply(lambda x: 1 if x else 0))
# plt.title("Histogram of target variable")
# plt.xlabel('value')
# plt.ylabel('number of occurrences')
# plt.show()
#
# plt.hist(df.innings1_runs)
# plt.title("Histogram of runs scored in 1st inning")
# plt.show()
#
# plt.hist(df.innings1_wickets)
# plt.title("Histogram of wickets from 1st inning")
# plt.show()
#
# plt.hist(df.innings1_overs)
# plt.title("Histogram of inning1_overs")
# plt.show()
#
# plt.hist(df.innings1_balls_bowled)
# plt.title("Histogram of innings1 balls bowled")
# plt.show()

combined.boxplot('date', 'target')
plt.show()

combined.boxplot('innings1_runs', 'target')
plt.show()

combined.boxplot('innings1_wickets', 'target')
plt.show()

combined.boxplot('innings1_overs', 'target')
plt.show()

combined.boxplot('innings1_balls_bowled', 'target')
plt.show()

classifiers = load_trained_models()

# based on scikit learn documentation on ROC Curves
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
for i in range(len(classifiers)):
    if classifiers[i][0] in ['svm']:
        # svm and logReg classifiers have better ROC curves given
        #   decision_functions.
        y_predicted_all = classifiers[i][1].decision_function(X_test)
    else:
        y_predicted_all = classifiers[i][1].predict_proba(X_test)[:, 1]
    y_predicted_actual = classifiers[i][1].predict(X_test)
    # print("num observations:", y_predicted.shape, X_test.shape)
    plt.figure(i)
    plt.title(classifiers[i][0] + "ROC Curve")
    fpr, tpr, _ = roc_curve(
        y_test,
        y_predicted_all)

    fpr1, tpr1, _ = roc_curve(
        y_test,
        y_predicted_actual)

    #print(thresholds)
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
    plt.savefig(join(data_folder, 'images', classifiers[i][0]+'.png'))
    plt.show()
    print(classifiers[i][0])
    print(confusion_matrix(y_test, y_predicted_actual))
    if classifiers[i][0] in ['knn', 'ranFor', 'svm']:
        print(classifiers[i][1][1].best_params_)

