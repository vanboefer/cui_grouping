from typing import Optional, List, Set, Hashable, TypeVar
import argparse
import pickle
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
from load_data import load_data


IndexKey = TypeVar('IndexKey', bound=Hashable)
PATH = Path('../data/groupings/')


class Groupings(object):
    """
    Groupings class
    ===============

    Attributes
    ----------
    ## INPUT
    name: str
        The name of the data used for grouping, e.g. all/ctgov/pubmed
    data: pd.DataFrame
        Dataframe containing the record id as index and the columns `disease_cuis` and `drug_cuis`.
    metric: {'cosine', 'jaccard'}
        The pairwise distance metric to be used. NOTE: 'jaccard' cannot be used on big datasets since this metric does not support sparse matrix inputs; this will throw a memory error.
    distance_threshold: float
        Threshold determining if two elements are to be considered similar.

    ## OUTPUT
    groups: pd.Series
        A series of all groups found within the data.
        - Index: group id.
        - Value: tuple of record id's belonging to the group.
    group_sizes: pd.Series
        A series of the size (number of items) of each group.
    supergroups: pd.Series
        A series of all supergroups (groups that are not subsets of other groups) found within the data.
        - Index: group id.
        - Value: tuple of record id's belonging to the group.
    supergroup_sizes: pd.Series
        A series of the size (number of items) of each supergroup.
    groups_per_record: pd.Series
        A series of all record id's and the groups to which they belong, if any.
        - Index: record id.
        - Value: list of group id's to which the record id belongs.
            None if record id is not associated with any groups.
    grouped_items: set
        A set of all items that belong to at least one group.

    Methods
    -------
    create_groups:
        Create the groups attribute by applying the pairwise distance metric and distance threshold.
    create_groups_chunked:
        Use chunks to create the groups attribute as in the create_groups method, in order to save memory.
    create_supergroups:
        Create the supergroups attribute by removing all subsets within groups.
    get_records_per_group:
        Return all record id's for one or more group id's.
    create_distance_matrix:
        Return a distance matrix.
    to_pickle:
        Pickle the object. The data is not stored with the pickled object.
    read_pickle:
        Read a pickled object from disk.
    """


    def __init__(
        self,
        name : str,
        data : pd.DataFrame,
        metric : str,
        distance_threshold : float,
    ) -> None:

        if not metric in ['cosine', 'jaccard']:
            raise ValueError(f"Metric '{metric}' not recognized; acceptable values are 'cosine' or 'jaccard'.")

        self.name = name
        self.data = data
        self.metric = metric
        self.sparse = metric == 'cosine'
        self.distance_threshold = distance_threshold


    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        del state['data']
        return state


    def __setstate__(self, state : dict) -> None:
        self.__dict__.update(state)
        self.data = None
        return None


    @property
    def preprocessed_data(self):
        "Preprocessed data: rows where both disease_cuis and drug_cuis are empty are removed."

        dis_crit = self.data.disease_cuis.apply(bool)
        drug_crit = self.data.drug_cuis.apply(bool)
        return self.data.loc[dis_crit | drug_crit]


    @property
    def groups(self):
        "All groups found within the data."

        if not hasattr(self, '_groups'):
            try:
                self.create_groups()
            except MemoryError:
                print('Insufficient memory; moving to chunked group creation.')
                self.create_groups_chunked()
        return pd.Series(self._groups, name='groups')


    @property
    def group_sizes(self):
        "Sizes (number of items) of all groups."
        return pd.Series(self._groups.apply(len), name='groups')


    @property
    def supergroups(self):
        "All supergroups (groups that are not subsets of other groups) within the data."

        if not hasattr(self, '_supergroups'):
            self.create_supergroups()
        return pd.Series(self._supergroups, name='supergroups')


    @property
    def supergroup_sizes(self):
        "Sizes (number of items) of all groups."
        return pd.Series(self._supergroups.apply(len), name='supergroups')


    @property
    def grouped_items(self):
        "All items that belong to at least one group."
        if not hasattr(self, '_grouped_items'):
            self._grouped_items = set(self.groups.explode().values)
        return self._grouped_items


    def create_groups(self) -> None:
        """
        (1) Apply the pairwise distance `metric` to sets of disease CUI's and drug CUI's in the `data`.
        (2) Mark as similar the record pairs where the distance is below the `distance_threshold`.
        (3) Form groups of records for which both the disease CUI's and the drug CUI's are similar.
        """

        def find_similar(label : str) -> pd.DataFrame:
            """
            (1) Represent the presence of cuis in a binary array.
            (2) Calculate pairwise distances between all records in the array.
            (3) Select record pairs with distance below the threshold.
            """
            binary = self.binarize(self.preprocessed_data[label], sparse_output=self.sparse)
            distance = pairwise_distances(binary, metric=self.metric)
            similar = distance < self.distance_threshold
            return pd.DataFrame(similar, columns=self.preprocessed_data.index)

        print('Creating groups; this might take a while...')

        sim_disease = find_similar('disease_cuis')
        sim_drug = find_similar('drug_cuis')

        sim_combined = sim_disease & sim_drug
        found_similarities = sim_combined.sum(axis=1) > 1

        self._groups = (
            sim_combined[found_similarities]
            .apply(lambda x: tuple(x.index[x]), 1)
            .drop_duplicates()
            .reset_index(drop=True)
        )
        return None


    def create_groups_chunked(self, **kwargs) -> None:
        """
        The processing is the same as in the create_groups method, but the pairwise distance calculations are done in chunks because of memory restrictions.

        Pass `n_jobs` and `working_memory` to kwargs to control chunked pairwise comparison.

        (1) Apply the pairwise distance `metric` to sets of disease CUI's and drug CUI's in the `data`.
        (2) Mark as similar the record pairs where the distance is below the `distance_threshold`.
        (3) Form groups of records for which both the disease CUI's and the drug CUI's are similar.
        """

        processed_chunks = []

        def gen_pairwise_distances(label, **kwargs):
            """
            (1) Represent the presence of cuis in a binary array.
            (2) Calculate pairwise distances between all records in the array.
            (3) Select record pairs with distance below the threshold.
            """
            def apply_threshold(chunk, _):
                similar = chunk < self.distance_threshold
                return pd.DataFrame(similar, columns=self.preprocessed_data.index)

            binary = self.binarize(self.preprocessed_data[label], sparse_output=self.sparse)
            return pairwise_distances_chunked(
                binary,
                metric=self.metric,
                reduce_func=apply_threshold,
                **kwargs,
            )

        print('Creating groups chunkwise; this might take a while...')

        zipped_distances = zip(
            gen_pairwise_distances('disease_cuis', **kwargs),
            gen_pairwise_distances('drug_cuis', **kwargs),
        )

        total_records = len(self.preprocessed_data)
        processed_records = 0
        for idx, (sim_disease, sim_drug) in enumerate(zipped_distances):
            sim_combined = sim_disease & sim_drug
            found_similarities = sim_combined.sum(axis=1) > 1

            processed_chunks.append(
                sim_combined[found_similarities]
                .apply(lambda x: tuple(x.index[x]), 1)
            )

            processed_records += len(sim_combined)
            print(
                datetime.now(),
                idx,
                processed_records,
                round(processed_records / total_records * 100, 1),
                found_similarities.sum(),
            )

        self._groups = (
            pd.concat(processed_chunks, ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )
        return None


    def create_supergroups(self):
        """
        (1) Sort groups by length.
        (2) For each group from largest to smallest, check for subsets and remove them.
        (3) Add all groups that are not subsets to supergroups.
        """

        print('Creating supergroups; this might take a while...')

        supergroups = pd.Series(dtype='object')
        groups = self.groups.copy()
        len_groups = groups.apply(len)
        groups = groups.reindex(len_groups.sort_values(ascending=False).index)
        while len(groups) > 0:
            group = set(groups.iloc[0])
            idx = groups.index[0]
            supergroups.loc[idx] = group
            groups = groups.drop(idx)

            test_groups = groups.copy()
            for k, v in test_groups.items():
                if group.issuperset(set(v)):
                    groups = groups.drop(k)

        self._supergroups = supergroups
        return None


    def get_records_per_group(self, *groups : IndexKey) -> Set[IndexKey]:
        "Return a set of all record id's for one or more group id's."

        selected_groups = set()
        for group in groups:
            selected_groups |= set(self.groups.loc[group])
        return selected_groups


    @property
    def groups_per_record(self):
        """
        Series containing all record id's and the group id's to which they belong (if any).
        """

        if not hasattr(self, '_groups_per_record'):
            self.create_groups_per_record()
        return self._groups_per_record


    def create_groups_per_record(self) -> None:
        def find_groups(
            rec_id : IndexKey
        ) -> Optional[List[IndexKey]]:
            rec_id_in_groups = self.groups.apply(lambda x: rec_id in x)
            found_groups = self.groups.index[rec_id_in_groups].to_list()
            return found_groups if found_groups != [] else None

        print('Creating groups per record; this might take a while...')
        self._groups_per_record = self.preprocessed_data.index.to_series().apply(find_groups)
        return None


    def to_pickle(self) -> None:
        """
        - Pickle the grouping and store it.
        - Filename is created from the name, the metric and the threshold (without decimal point, e.g. 02) given when initialized.
        - Data is not saved.
        """

        threshold = str(self.distance_threshold).replace('.', '')
        path = PATH / '_'.join([self.name, self.metric, threshold])
        with open(path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(self, f)
        return None


    @staticmethod
    def read_pickle(
        name,
        metric,
        threshold,
        data=None,
    ):
        "Read a previously stored grouping. Call grouping by passing `name`, `metric` and `threshold` (without decimal point, e.g. 02)."

        threshold = str(threshold).replace('.', '')
        path = PATH / '_'.join([name, metric, threshold])
        with open(path.with_suffix('.pkl'), 'rb') as f:
            groupings = pickle.load(f)
        if data is not None:
            groupings.data = data
        return groupings

    @staticmethod
    def view_saved_groupings():
        for file in PATH.glob('*.pkl'):
            print(file.relative_to(PATH))


    @staticmethod
    def binarize(array, sparse_output=False):
        """
        Return a (samples x cuis) binary matrix indicating the presence of a cui.
        """

        binarizer = MultiLabelBinarizer(sparse_output=sparse_output)
        return binarizer.fit_transform(array)


    def create_distance_matrix(self, label : str) -> pd.DataFrame:
        """
        (1) Represent the presence of drug_cuis/disease_cuis (according to `label`) in a binary array.
        (2) Calculate pairwise distances between all records in the array.
        (3) Return a dataframe of the distances.
        """

        binary = self.binarize(self.preprocessed_data[label], sparse_output=self.sparse)
        distance = pairwise_distances(binary, metric=self.metric)
        return pd.DataFrame(
            distance,
            index=self.preprocessed_data.index,
            columns=self.preprocessed_data.index,
        )


if __name__ == '__main__':

    ############## PARAMETERS ##############
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--name', default='sample_data')
    argparser.add_argument('--metric', default='cosine')
    argparser.add_argument('--distance_threshold', default=0.4)
    argparser.add_argument('--working_memory', default=8192)
    args = argparser.parse_args()

    ############## LOAD DATA ##############
    sample_data = load_data()
    ema = pd.read_pickle('../data/ema_cuis.pkl')

    ############## COMBINE DATA ##############
    def combine_dfs(*args, keys=None):
        return pd.concat([*args], join='inner', keys=keys)

    sample_data = combine_dfs(ema, sample_data, keys=['ema', 'sample_data'])
    print(f"Combined data ('sample_data') includes {sample_data.shape[0]} records (ctgov, pubmed, ema)")

    ############## GROUP ##############
    name = args.name
    metric = args.metric
    distance_threshold = float(args.distance_threshold)
    working_memory = int(args.working_memory)

    g = Groupings(name, sample_data, metric, distance_threshold)

    # create groups
    g.create_groups_chunked(working_memory=working_memory)

    #create supergroups
    g.create_supergroups()

    ############## PICKLE ##############
    g.to_pickle()
    print(f"Pickled Groupings object is saved at ../data/groupings/{'_'.join([name, metric, str(distance_threshold)])}")
