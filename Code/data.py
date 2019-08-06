import pandas as pd
import os

data_folder = "/home/zbuckley/Dropbox/"


def get_data(directory):
    return pd.read_csv(os.path.join(directory, "t20_matches.csv")), \
           pd.read_csv(os.path.join(directory, "t20_series.csv"))

#def clean_matches(matches):

