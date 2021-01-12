import argparse
import requests
import json
import logging
import datetime
import time
import pandas as pd
from pathlib import Path


# logging
logging.basicConfig(
    filename='fail_log',
    level=logging.WARNING)
log = logging.getLogger(__name__)


URL_BERN = 'https://bern.korea.ac.kr/plain'


def bernotate_df(df, path, fail_limit):
    """
    Annotate the text in the `df` with BERN and save the results in json files (a file per record) in the directory specified by `path`.

    Parameters
    ----------
    df: pandas DataFrame
        dataframe that contains the text for annotation in a column called 'all_text_clean' and a unique record id as the index
    path: str
        path to the directory where the output json files are stored
    fail_limit: int
        the number of consecutive errors (when sending a request for BERN annotation) after which the annotation procedure breaks

    Returns
    -------
    None
    """

    start       = time.time()
    length      = len(df)
    fail_count  = 0
    total_fails = 0

    for cnt, row in enumerate(df.itertuples()):
        idx = row.Index
        text = row.all_text_clean
        display_status(cnt, idx, length)
        status = bernotate_text(idx, text, path)

        if not status:
            fail_count  += 1
            total_fails += 1
            if fail_count > fail_limit and fail_limit > 0:
                print(f"FAILED {fail_count} TIMES. BREAKING DOWN PROCEDURE.")
                break

    exc_time = str(datetime.timedelta(seconds=time.time() - start))
    print(
        "BERN annotation completed; execution time: "
        f"{exc_time} hh:mm:ss; "
        f"{cnt + 1} records processed; "
        f"{total_fails} records failed."
    )


def bernotate_text(idx, text, path):
    """
    Annotate `text` with BERN and store the results in a json file (file name defined by `idx`) in the directory specified by `path`.

    Parameters
    ----------
    idx: {str, int}
        unique identifier of the processed record
    text: str
        text to be annotated
    path: str
        path to the directory where the output json file is stored

    Returns
    -------
    True: bool
        if annotation succeeded
    False: bool
        if annotation failed
    """

    try:
        data = {'sample_text': text}
        response = requests.post(URL_BERN, data=data)
        response.raise_for_status()
        print('| status_code: ', response.status_code)

        parsed = response.json()
        file = path / f"{idx}.json"
        with file.open('w') as f:
            json.dump(parsed, f)
        return True

    except requests.exceptions.ConnectionError:
        log.warning(f"{idx} failed on ConnectionError")
        print('| connection failed')
        return False

    except requests.exceptions.HTTPError:
        log.warning(f"{idx} failed on HTTPError")
        print('| http failed')
        return False

    except json.decoder.JSONDecodeError:
        log.warning(f"{idx} failed on JSONDecodeError")
        print('| json decoder failed')
        return False


def display_status(cnt, idx, length):
    """
    Displays progress of the annotation (see the function `bernotate_df`).
    """

    percentage = f"({((cnt + 1) / length) * 100:.1f} %)"
    print(
        f"{cnt + 1:0{len(str(length))}d} / {length} "
        f"{percentage:>9} | "
        f"Processing record {idx}", end=' ', flush=True,
    )


if __name__ == '__main__':

    """
    Annotate text with BERN and save the results in json files (a file per record) in the directory specified by the '--path_out' parameter.

    The text used for annotation is found under the 'all_text_clean' column in the pickled dataframe specified by the '--data' parameter.

    To determine the batch, use the parameters '--start' and '--end';
    this refers to the row index in the pickled dataframe specified by the '--data' parameter.

    The '--fail_limit' parameter determines the number of consecutive errors
    (when sending a request for BERN annotation) after which the annotation procedure breaks.
    """

    ############## PARAMETERS ##############
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data', default='../data/data_sample.pkl')
    argparser.add_argument('--path_out', default='../data/bern_json/')
    argparser.add_argument('--start', default=0)
    argparser.add_argument('--end', default=200)
    argparser.add_argument('--fail_limit', default=5)
    args = argparser.parse_args()

    path = Path(args.path_out)
    path.mkdir(exist_ok=True, parents=True)

    ############## LOAD DATA ##############
    df = pd.read_pickle(args.data)

    ############## ANNOTATE with BERN ##############
    batch = df.iloc[int(args.start):int(args.end)+1]

    # check which studies in the batch are already annotated and exclude them
    print('Initial number of records: ', len(batch))
    idx = [f.stem for f in path.glob('*.json')]
    batch = batch[~batch.index.isin(idx)]
    print('Number of records to be processed: ', len(batch))

    bernotate_df(batch, path, fail_limit=int(args.fail_limit))
