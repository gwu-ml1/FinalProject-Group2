import pandas as pd
import numpy as np
import os

data_folder = "../Data/" #path should work if code is executed from inside Code folder


def get_matches(directory=data_folder):
    return pd.read_csv(os.path.join(directory, "t20_matches.csv"),
                       index_col=0)


# need to verify this with vibhu... pretty sure it'll work
#   some quick testing... seems find. need to clean data a bit first.
def oversToBallsBowled(overs):
    from math import floor
    runsPerOver = 6
    return floor(overs)*runsPerOver + (overs-floor(runsPerOver))*10


# function to drop abandoned matches
def dropAbandonedMatches(df):
    # drop matches abandoned with a toss
    df.drop(df[df['winner'] == "No result (abandoned with a toss)"].index, inplace=True)
    # drop matches abandoned with no balls bowled
    df.drop(df[df['result'] == "Match abandoned without a ball bowled"].index, inplace=True)
    # canceled instead of abandoned... why
    df.drop(df[df['result'] == "Match cancelled without a ball bowled"].index, inplace=True)
    # Matches with No Results
    df.drop(df[df['winner'] == "No result"].index, inplace=True)
    df.drop(df[df.result.isna() == True].index, inplace=True)
    # and two special cases... with more unique abandonment text...
    df.drop([383219, 514874], inplace=True)


# helper that computes percent reduced between shape tuples
def percentReduced(prevSize, newSize, axis=0):
    return ((prevSize[axis]-newSize[axis])/prevSize[axis])*100


# drop columns we shouldn't use given our problem constraints
def dropCols(df):
    # drop original text-based columns that data set creator
    #   apparently parsed to build the rest of the columns.
    # reference: https://www.kaggle.com/imrankhan17/t20matches
    df.drop(['match_details', 'result', 'scores'], axis=1, inplace=True)
    # drop columns not available at the end of the 1st inning
    df.drop(['winner', 'win_by_runs', 'win_by_wickets', 'balls_remaining',
             'innings2_runs', 'innings2_wickets', 'innings2_overs_batted',
             'D/L_method', 'innings2_overs'], axis=1, inplace=True)

# this code originally provided by Vibhu.
# It was since modified as we went along.
def clean_matches(df=get_matches()):
    size0 = df.shape
    print("Initial dataset dimensions:", size0)

    # deleting games with no result
    dropAbandonedMatches(df)

    size1 = df.shape
    print("After Dropping abandoned matches:")
    print("\tdataset dimensions:", size1)
    print("\tNumber of observations reduced by: ", percentReduced(size0, size1), "percent.")

    # create target feature; is true if 1st inning team wins
    #  this will override a target feature already provided, but
    #  it's a column we can't use anyways. Note: this does mean
    #  we aren't removing the original target column any longer
    #  in dropCols below
    df['target'] = df.innings1 == df.winner

    # drop columns for various reasons
    # have to build the target column first!!!!
    dropCols(df)

    size2 = df.shape
    print("After adding innings1_win, and dropping various columns:")
    print("\tdataset dimensions:", size2)
    print("\tNumber of features reduced by: ", percentReduced(size0, size2, axis=1), "percent.")

    # transform innings1_overs_batted to innings1_balls_bowled? TODO: verify terminology with Vibhu
    df['innings1_balls_bowled'] = df['innings1_overs_batted'].apply(oversToBallsBowled)
    df.drop('innings1_overs_batted', axis=1, inplace=True)
    return df


df = clean_matches()
df.to_csv(os.path.join(data_folder, 'cleaned.csv'))