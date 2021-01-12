"""
Functions for loading ctgov and pubmed data into pd DataFrame's.
Records where both disease_cuis and drug_cuis are empty are removed (their total number is printed).
"""


import pickle
import pandas as pd
from pathlib import Path


def load_data(
    cuis_dir : str = '../data/cuis/',
) -> pd.DataFrame:
    """
    - Read and combine pickled dataframes in the directory defined by `cuis_dir`.
    - Remove records where both disease_cuis and drug_cuis are empty (their number is printed).
    """

    data = (
        create_df_from_pickles(cuis_dir)
        .pipe(nan_to_emptyset)
        .pipe(remove_no_cuis, name='CTgov')
    )

    print(f"{data.shape[0]} records are loaded")
    return data


def create_df_from_pickles(path):
    if path == None:
        return None
    elif isinstance(path, str):
        path = Path(path)
    return pd.concat([pd.read_pickle(f) for f in path.glob('*.pkl')])

def fill_nan(x):
    return {} if x != x else x

def nan_to_emptyset(df):
    df.disease_cuis = df.disease_cuis.apply(fill_nan)
    df.drug_cuis = df.drug_cuis.apply(fill_nan)
    return df

def remove_no_cuis(df, name):
    dis_crit = df.disease_cuis.apply(bool)
    drug_crit = df.drug_cuis.apply(bool)
    output = df.loc[dis_crit | drug_crit]
    print(f"{len(df) - len(output)} {name} records are removed where both disease_cuis and drug_cuis are empty.")
    return output
