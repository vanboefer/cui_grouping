import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from quickumls import QuickUMLS


def process_batch(filepath, id):
    """
    Process a pkl file containing a dataframe of BERN annotations; each BERN entity has its own row (i.e. there are multiple rows per record). Return a dataframe where each unique record has one row and drug and disease entities associated with the record are grouped under respective columns.

    Parameters
    ----------
    filepath: str
        path to pkl file with a dataframe of BERN annotations (entity per row)
    id: str
        name of the column containing the unique record identifiers, by which the entities should be grouped

    Returns
    -------
    processed_batch: pd.DataFrame
        dataframe where each unique record has one row, all disease entities associated with the record are in an array in the `ent_text_disease` column and all drug entities associated with the record are in an array in the `ent_text_drug` column
    """

    batch = pd.read_pickle(filepath)
    processed_batch = pd.DataFrame(index=batch[id].unique())

    batch['entity_text'] = batch['entity_text'].str.lower()
    dis_series = batch.loc[(batch.entity_type == 'disease')].groupby(id).entity_text.unique()
    drug_series = batch.loc[(batch.entity_type == 'drug')].groupby(id).entity_text.unique()

    processed_batch = processed_batch.join(dis_series)
    processed_batch = processed_batch.join(drug_series, rsuffix='_drug')
    processed_batch.columns = ['ent_text_disease', 'ent_text_drug']
    return processed_batch


def apply_QuickUMLS(txt_array, matcher):
    """
    Apply QuickUMLS on the text in `txt_array` and return the set of identified UMLS CUIs.

    Parameters
    ----------
    txt_array: {str, np.ndarray}
        the text to process with QuickUMLS
    matcher: QuickUMLS object

    Returns
    -------
    cuis: set
        set of cuis identified by QuickUMLS
    """

    if isinstance(txt_array, str):
        txt_array = np.array(txt_array)

    if txt_array is np.nan:
        return np.nan
    cuis = set()
    for match in matcher.match(txt_array, best_match=True, ignore_syntax=False):
        best_match = match[0]
        cuis.add(best_match['cui'])
    return cuis


if __name__ == '__main__':

    """
    Process pkl files containing dataframes with detected BERN entities (of types 'drug' and 'disease') per CTgov/PubMed record (directory is specified by the '--path_in' parameter).

    Return pkl files of the dataframes with two added columns containing the CUIs identified by QuickUMLS (directory is specified by the '--path_out' parameter).
    """

    ############## PARAMETERS ##############
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--path_in', default='../data/bern_df/')
    argparser.add_argument('--path_out', default='../data/cuis/')
    args = argparser.parse_args()

    ############## INSTANTIATE QuickUMLS ##############
    sem_diseases = ['T020', 'T190', 'T049', 'T019', 'T047', 'T050', 'T033', 'T037', 'T048', 'T191', 'T046', 'T184']
    sem_drugs = ['T116', 'T195', 'T123', 'T122', 'T103', 'T120', 'T104', 'T200', 'T196', 'T126', 'T131', 'T125', 'T129', 'T130', 'T197', 'T114', 'T109', 'T121', 'T192', 'T127']
    sem_dis_drug = sem_diseases + sem_drugs
    data_dir = '../data/quickUMLS_eng'

    matcher = QuickUMLS(quickumls_fp=data_dir, accepted_semtypes=sem_dis_drug)

    ############## PROCESS ##############
    path_in = Path(args.path_in)
    path_out = Path(args.path_out)
    path_out.mkdir(exist_ok=True, parents=True)

    for file in path_in.glob('*.pkl'):
        batch = process_batch(file, 'idx')
        batch_name = file.stem
        batch['disease_cuis'] = batch['ent_text_disease'].apply(apply_QuickUMLS, args=(matcher,))
        batch['drug_cuis'] = batch['ent_text_drug'].apply(apply_QuickUMLS, args=(matcher,))
        batch.to_pickle(f"{path_out}/{batch_name}.pkl")
        print(f"{batch_name} is processed and saved.")
