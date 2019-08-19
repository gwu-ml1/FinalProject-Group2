import pandas as pd
from common_utils import load_cleaned
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import cohen_kappa_score, accuracy_score, make_scorer

def mlp(X_train, y_train):

    # Code below heavily inspired by sklearn documentation, in particular:
    # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#sphx-glr-auto-examples-compose-plot-column-transformer-mixed-types-py
    # building ingest pipeline using sklearn
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

    # unsure if this is a valid approach... but due to runtime constraints...
    # 1) set gridsearchCV to check between logistic, tanh, and relu activation functions
    #  logistic performed best with acc. of .68, and cohen's kappa of .35 on test set
    # 2) now running gridsearchCV to check between different solvers
    #  sgd performed best with acc. of .69 and cohen's kappa of .38.
    # 3) now running gridsearchCV to check between learning rates
    #  constant performed best with acc. of .69 and cohen's kappa of .37
    # 4) now lets play with number of neurons, number of layers
    #  - default number of neurons is 100, with only one hidden layer
    #  - lets try upping that to half the number of features: 340 teams times 4 columns with teams as entries
    #      gives us 1360+515('round' columns... prolly should drop this)+5(continuous). so lets try a single
    #      layer with n/2, n/4, n/8 neurons, and see what happens. 940, 470, 235
    # Starting to see why keras GPU acceleration on this stuff is helpful... this is taking longer than i expected.
    #   to be fair it's training the neural network 15 times...
    # OK... got tired of waiting on this... so I'm going to setup hidden layers as (200,), and
    #   call it our sklearn based mlp model. Then move on to building an mlp model using keras.
    # Curiously, I noted that attempting to add more hidden layers of the same size, didn't help our classification
    #   at all, putting us in the range of 52%, which is roughly equivlent to the NullAccuracy rate. it also takes much
    #   longer to train on just (200,) vs (200, 200) or (200,200,200)... not sure I understand why.


    mlpModel = MLPClassifier(
            activation='logistic',
            solver='sgd',
            learning_rate='adaptive',
            hidden_layer_sizes=(200,),
            max_iter=500)

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
    print("finished training Sklearn based Multi-layer perceptron")
    print("\taccuracy:", accuracy_score(y_test, y_predict))
    print("\tcohen's kappa:", cohen_kappa_score(y_test, y_predict))

