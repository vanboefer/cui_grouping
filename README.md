*CUI_grouping*
==============
The scripts in this repo are used to group biomedical records from different sources based on the drug and disease entities mentioned in them.

The method extracts and normalizes drug and disease names from the free text of various documents. It then groups together records that mention the same diseases and drugs.

The three data sources from which records are grouped are:

- CTgov: a register of clinical trials ([https://clinicaltrials.gov/](https://clinicaltrials.gov/))
- PubMed: a database of biomedical publications ([https://pubmed.ncbi.nlm.nih.gov/](https://pubmed.ncbi.nlm.nih.gov/))
- EMA drug register ([https://www.ema.europa.eu/en/medicines](https://www.ema.europa.eu/en/medicines))

A sample of 100 CTgov records, 100 PubMed records and 72 EMA records can be found in the [data](data/) folder.

## Step 1: BERN annotation
[BERN](https://github.com/dmis-lab/bern) is a state-of-the-art biomedical named entity recognition tool, which detects entities of types *disease*, *drug*, *gene*, *species* and *mutation*. Please refer to the paper by [Kim et al. (2019)](https://ieeexplore.ieee.org/document/8730332) for further information.

The [`bernotate.py`](code/bernotate.py) script creates a json file per record with all the BERN-detected biomedical entities.

## Step 2: Parse BERN json files
The [`bernparse.py`](code/bernparse.py) script parses the json files created in [Step 1](#step-1-bern-annotation) and stores the detected entities in pickled dataframes. The processing is done in batches of 10,000 json files.

## Step 3: Map BERN entities to UMLS CUIs
The [UMLS metathesaurus](https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/index.html) maps synonymous medical terms to a *concept unique identifier* (CUI); for example, the disease terms *Hb-SS disease*, *Herrick syndrome* and *sickle cell anemia* (which all refer to the same disease) are mapped to the CUI C0002895. The tool used for mapping is [QuickUMLS](https://github.com/Georgetown-IR-Lab/QuickUMLS); please refer to the paper by [Soldaini and Goharian (2016)](http://ir.cs.georgetown.edu/downloads/quickumls.pdf) for further information.

The [`get_cuis.py`](code/get_cuis.py) script processes the pickled dataframes created in [Step 2](#step-2-parse-bern-json-files). The *disease* and *drug* entities in the dataframes are mapped to CUIs. The results are stored in pickled dataframes.

**NOTE**: To run QuickUMLS, you will need to obtain a license from the National Library of Medicine and download the UMLS files. See instructions and links [here](https://github.com/Georgetown-IR-Lab/QuickUMLS). The script assumes that a folder called `quickUMLS_eng`, which contains the UMLS data for English, is located in the [data](data/) folder.

## Step 4: Grouping
The script [`groupings.py`](code/groupings.py) groups the records based on their disease and drug CUI's using a similarity measure and a threshold (defined by the available parameters; see the script).

The results of grouping the sample data using cosine similarity and a distance threshold of 0.4 (i.e. similarity threshold 0.6) can be found in [data/groupings](data/groupings/); these results are examined in the notebook [`results_explore.ipynb`](code/results_explore.ipynb).
