import pandas as pd
import numpy as np
import os

data_folder = "../Data/" #path should work if code is executed from inside Code folder


def get_matches(directory=data_folder):
    return pd.read_csv(os.path.join(directory, "t20_matches.csv"), index_col=0)


# this code originally provided by Vibhu. Modified to fit in context.
def clean_matches(df=get_matches()):
    # deleting games with no result
    indexNames = df[df['winner'] == 'No result (abandoned with a toss)'].index
    df.drop(indexNames, inplace=True)

    indexNames1 = df[df['winner'] == 'No result'].index
    df.drop(indexNames1, inplace=True)

    # delete the column date

    df = df.drop('date', axis=1)

    # delete the column round

    df = df.drop('round', axis=1)

    # delete the rows where games were calculated by D/L method
    indexNames3 = df[df['D/L_method'] == 1].index
    df.drop(indexNames3, inplace=True)

    # Converting match Id and Series ID to categorical

    df['series_id'] = pd.Categorical(df.series_id)
    # df['match_id'] = pd.Categorical(df.match_id)
    # delete the kenya series for lack of data: series ID
    indexNames3 = df[df['series_id'] == '691361'].index
    df.drop(indexNames3, inplace=True)

    # deleting rows without data

    df = df.dropna(subset=['result'])

    # drop rows where matches have been a tie
    df.drop(df.loc[df['innings1_runs'] == df['innings2_runs']].index, inplace=True)

    # tried converting to base six not valid as
    df['innings1_overs_batted'] = df['innings1_overs_batted'].apply(lambda x: x * 6)
    df['innings2_overs_batted'] = df['innings2_overs_batted'].apply(lambda x: x * 6)

    np.ceil(df.innings1_overs_batted)
    print(df)

clean_matches()