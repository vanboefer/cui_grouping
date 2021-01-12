import argparse
import json
import pandas as pd
from pathlib import Path


def parse_file(filepath):
    """
    Process a json file with BERN annotations and return a list of dictionaries containing info about all entities in the file.

    Parameters
    ----------
    filepath: str
        path to json file with BERN annotations

    Returns
    -------
    rows: list
        list of dicts; each dict contains information about one entity identified by BERN
    """

    with open(filepath) as jfile:
        json_data = json.load(jfile)

    rows = []
    for el in json_data['denotations']:
        row_dict = {}
        row_dict['idx'] = filepath.stem
        row_dict['annotated_text'] = json_data['text']

        # retrieve entity span and text
        start_span = el['span']['begin']
        end_span = el['span']['end']
        row_dict['span_begin'] = start_span
        row_dict['span_end'] = end_span
        row_dict['entity_text'] = json_data['text'][start_span:end_span]

        # retrieve entity type
        row_dict['entity_type'] = el['obj']

        # retrieve entity id
        row_dict['entity_IDs'] = ','.join(el['id'])
        rows.append(row_dict)
    return rows


if __name__ == '__main__':

    """
    Extract all entities from a directory of json files containing BERN annotations (directory is specified by the '--path_json' parameter) and store them in a pickled dataframe.

    The processing is done in batches of 10,000 and the batches are pickled in the directory specified by the '--path_pkl' parameter.

    The output dataframe contains the following columns:
    - idx (refers to the pmid/nct_id)
    - annotated_text
    - span_begin
    - span_end
    - entity_text
    - entity_type
    - entity_IDs
    """

    ############## PARAMETERS ##############
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--path_json', default='../data/bern_json/')
    argparser.add_argument('--path_pkl', default='../data/bern_df/')
    args = argparser.parse_args()

    ############## PARSE JSON ##############
    path_json = Path(args.path_json)
    path_pkl = Path(args.path_pkl)
    path_pkl.mkdir(exist_ok=True, parents=True)

    rows = []
    for cnt, f in enumerate(path_json.glob('*.json')):
        progress = cnt % 10000
        batch = cnt // 10000
        if progress == 9999:
            pd.DataFrame(rows).to_pickle(f"{path_pkl}/batch_{batch}.pkl")
            rows = []
        rows.extend(parse_file(f))
    else:
        pd.DataFrame(rows).to_pickle(f"{path_pkl}/batch_{batch}.pkl")
