import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split
from joblib import load
from os import path

data_folder = path.join('..', 'Data')  # path should work so long as code is executed from Code directory


def load_cleaned(dir=data_folder):
    df = pd.read_csv(path.join(dir, "cleaned.csv"),
                     index_col=0)

    # setup common categorical dtype for teams.. then apply to all the teams features
    teams = df.home.append(
        df.away.append(
            df.innings1.append(
                df.innings2, ignore_index=True),
            ignore_index=True),
        ignore_index=True).dropna().unique()
    #  340
    print("total number of teams:", teams.shape)
    teams_catDType = CategoricalDtype(categories=teams, ordered=False)
    df['home'] = df.home.astype(teams_catDType)
    df['away'] = df.away.astype(teams_catDType)
    df['innings1'] = df.innings1.astype(teams_catDType)
    df['innings2'] = df.innings2.astype(teams_catDType)

    # venue should be categorical
    df['venue'] = df.venue.astype('category')

    # may want to drop round... let's try and do categorical though?
    df['round'] = df['round'].astype('category')
    print("entries in round:", df['round'].unique().shape)

    # date as date type
    df['date'] = pd.to_datetime(df['date'])

    # these should be ints
    df['innings1_runs'] = df.innings1_runs.astype(int)
    df['innings1_overs'] = df.innings1_overs.astype(int)
    df['innings1_wickets'] = df.innings1_wickets.astype(int)
    df['innings1_balls_bowled'] = df.innings1_balls_bowled.astype(int)

    # setup test/train split
    #  first need to separate the target column from the predictors/features
    target = df.target
    df.drop(['target'], inplace=True, axis=1)

    # can't imagine series_id being useful here... prolly drop or use it to pull other features?
    df.drop(['series_id'], inplace=True, axis=1)

    # https://stackoverflow.com/questions/16453644/regression-with-date-variable-using-scikit-learn
    # ran into a problem on date objects when training the model...
    # Choosing to turn the dates into ordinal values instead.
    df['date'] = df.date.apply(lambda x: x.toordinal())

    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


# Now I'm attempting to find 1 'ring' to rule them all.
#  by binding them... in the darkness... lol
def filepath(filename):
    return path.join(data_folder, 'persisted_models', filename)


def load_trained_models():
    return [('bayes', load(filepath('bernoulliBayes.joblib'))),
            ('knn', load(filepath('knn.joblib'))),
            ('logReg', load(filepath('logReg.joblib'))),
            ('ranFor', load(filepath('ranFor.joblib'))),
            ('sklearn_mlp', load(filepath('sklearn_mlp.joblib'))),
            ('svm', load(filepath('svm.joblib')))]


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_cleaned()
    print("X_train dimensions:", X_train.shape)
    print("X_test dimensions", X_test.shape)
    print("y_train dimensions", y_train.shape)
    print("y_test dimensions", y_test.shape)