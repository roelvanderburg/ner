# Automatic-NER tagging for names

## Contents
This repository is divided into three main folders being: pre-processing, matching and classifying.
1. Pre-processing handles the loading of json files, extracting the right columns to perform matching and saving the data as a pickled pandas matrix.
2. The matching directory contains a class to perform fuzzy string matching to generate a dataset.  
3. The classifying folder contains training file that generates a SVM to classify two-grams to either a name or no-name match.

4. The start directory contains a start.py script to initiate the pipeline.

<br>

### Pre-Processing

- **get_json.py**: handles the loading of json files, extracting the right columns to perform matching and saving the data as a pickled pandas matrix 
- **request_name_from_bsn.py**: contains a helper function for get_json to request information from the database given a bsn 

Gebruik van command line interface get_json.py

```
usage: get_json.py [-h] filepath save_file_name

Process json files for matching

positional arguments:
  filepath				filepath to the json files that need to be processed
  save_file_name			filename for the pd.DataFrame to be pickled

optional arguments:
  -h, --help            show this help message and exit
```

### Matching

- **n_gram_matching.py**: performs the fuzzy matching to generate a dataset, dependent on how many n_grams we want to try to match on (default=7).
- **update_files.py**:  python file contains a function to load in the original data, update the annotation field with the found matches (for viewing in the front end) and writes it out to the desired directory. 

### Classifying

- **train.py**: contains a train.py that generates a SVM to classify two-grams to either a name or no-name match.

This SVM is trained on a set of 28 features describing the two-gram. These include the presence of upper case and lower case characters, punctuation or the
length of each word itself. Some of these features are normalized and taken over parts of the two-gram (e.g amount of upper case characters in the first word of the two gram). This way the classifier ends up with 28 features.

Below, the results presented for the SVM classifier predicting on the hold out set of ~ 7200 two-grams. 

Confusion matrix             |  Normalized confusion matrix
:-------------------------:|:-------------------------:
<img src="https://devtools.belastingdienst.nl/bitbucket/projects/COOP/repos/ner-tag-roel/raw/classifying/images/confusion_matrix.png?at=refs%2Fheads%2Fmaster" alt="Confusion Matrix" width="420" height="310">  |  <img src="https://devtools.belastingdienst.nl/bitbucket/projects/COOP/repos/ner-tag-roel/raw/classifying/images/confusion_matrix_normalized.png?at=refs%2Fheads%2Fmaster" alt="Confusion Matrix" width="420" height="310">

